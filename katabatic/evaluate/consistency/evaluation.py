import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from katabatic.evaluate.base_evaluation import Evaluation


class ConsistencyEvaluation(Evaluation):
    """
    Consistency dimension: measures whether synthetic data is internally
    coherent and preserves the structural patterns of the real data.

    Metrics
    -------
    - Discriminator Score: a Random Forest classifier is trained to
      distinguish real rows (label=1) from synthetic rows (label=0).
      If the classifier cannot do better than ~50% accuracy, the synthetic
      data is statistically indistinguishable from real. Accuracy above 70%
      is a red flag — the model is generating obviously fake data.
      Score = max(0, 1 - (accuracy - 0.5) * 2), so 0.5 accuracy -> score 1.0,
      1.0 accuracy -> score 0.0.

    - Constraint Violation Rate: checks user-defined per-column bounds
      (min, max) against the synthetic data. None means unbounded on that
      side. Reports % of rows violating each constraint and an overall rate.
      Score = 1 - overall_violation_rate.

    - Feature Importance Spearman Correlation: trains a Random Forest on
      real data and on synthetic data separately (using the target column),
      extracts the feature importance ranking from each, and computes the
      Spearman rank correlation. A score close to 1 means the synthetic
      data preserves the same predictive structure as the real data.

    consistency_score is the mean of the three component scores.
    If no constraints are provided, consistency_score is the mean of the
    discriminator and Spearman scores only.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset including the target column.
    synthetic_data : pd.DataFrame
        Synthetic dataset with the same schema.
    target_col : str
        Name of the target column (needed for feature importance).
    constraints : dict[str, tuple], optional
        Per-column bounds: {col_name: (min_value, max_value)}.
        Use None for an unbounded side, e.g. {'age': (0, None)}.
    n_folds : int
        CV folds for the discriminator (default: 5).
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str,
        constraints: dict = None,
        n_folds: int = 5,
    ):
        super().__init__(real_data, synthetic_data)
        self.target_col = target_col
        self.constraints = constraints or {}
        self.n_folds = n_folds

    def evaluate(self) -> dict:
        discriminator_acc = self._compute_discriminator()
        constraint_results = self._compute_constraint_violations()
        spearman_score = self._compute_feature_importance_spearman()

        # Component scores in [0, 1].
        # abs() is required: accuracy below 0.5 means the RF distinguishes real from
        # synthetic but with inverted labels — still a detectable model. Without abs,
        # acc=0.1 gives a perfect 1.0 instead of the correct 0.2.
        disc_score = round(max(0.0, 1.0 - abs(discriminator_acc - 0.5) * 2), 4)
        constraint_score = (
            round(1.0 - constraint_results["overall_violation_rate"], 4)
            if constraint_results
            else None
        )
        spearman_val = spearman_score if spearman_score is not None else None

        active = [
            s for s in [disc_score, constraint_score, spearman_val] if s is not None
        ]
        consistency_score = round(float(np.mean(active)), 4) if active else 0.0

        results = {
            "discriminator_accuracy": round(discriminator_acc, 4),
            "discriminator_score": disc_score,
            "constraint_violations": constraint_results,
            "constraint_score": constraint_score,
            "feature_importance_spearman": round(spearman_val, 4)
            if spearman_val is not None
            else None,
            "consistency_score": consistency_score,
        }

        self._print_summary(results)
        return results

    def _compute_discriminator(self) -> float:
        """
        Combine real (1) and synthetic (0) rows, run k-fold CV with RF.
        Returns mean accuracy across folds.
        """
        real_enc = self._encode_dataframe(
            self.real_data.copy(), reference_df=self.synthetic_data
        )
        synth_enc = self._encode_dataframe(
            self.synthetic_data.copy(), reference_df=self.real_data
        )

        real_enc["_label"] = 1
        synth_enc["_label"] = 0

        combined = pd.concat([real_enc, synth_enc], ignore_index=True)

        # Feature columns only — exclude the label marker AND the target column.
        # Including the target lets the discriminator trivially distinguish real from
        # synthetic via target-distribution differences, which are already captured by
        # the Utility dimension and would unfairly penalise Consistency.
        feature_cols = [
            c for c in real_enc.columns if c != "_label" and c != self.target_col
        ]
        X = combined[feature_cols].values
        y = combined["_label"].values

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))

    def _compute_constraint_violations(self) -> dict:
        if not self.constraints:
            return {}

        results = {}
        n_synth = len(self.synthetic_data)
        violated_any = np.zeros(n_synth, dtype=bool)

        for col, (col_min, col_max) in self.constraints.items():
            if col not in self.synthetic_data.columns:
                continue

            vals = self.synthetic_data[col]
            violation_mask = np.zeros(n_synth, dtype=bool)

            if col_min is not None:
                violation_mask |= (vals < col_min).values
            if col_max is not None:
                violation_mask |= (vals > col_max).values

            rate = round(float(violation_mask.sum()) / n_synth, 4)
            results[f"{col} [{col_min}, {col_max}]"] = rate
            violated_any |= violation_mask

        results["overall_violation_rate"] = round(
            float(violated_any.sum()) / n_synth, 4
        )
        return results

    def _compute_feature_importance_spearman(self):
        """
        Train RF on real and synthetic data separately using target_col.
        Compare feature importance rankings via Spearman correlation.
        """
        if self.target_col not in self.real_data.columns:
            return None

        feature_cols = [c for c in self.real_data.columns if c != self.target_col]
        if not feature_cols:
            return None

        X_real = self._encode_dataframe(
            self.real_data[feature_cols].copy(),
            reference_df=self.synthetic_data[feature_cols],
        ).values
        y_real = self.real_data[self.target_col].values

        X_synth = self._encode_dataframe(
            self.synthetic_data[feature_cols].copy(),
            reference_df=self.real_data[feature_cols],
        ).values
        y_synth = self.synthetic_data[self.target_col].values

        clf_real = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_synth = RandomForestClassifier(n_estimators=100, random_state=42)

        clf_real.fit(X_real, y_real)
        clf_synth.fit(X_synth, y_synth)

        imp_real = clf_real.feature_importances_
        imp_synth = clf_synth.feature_importances_

        # Rank importances (rank 1 = most important)
        rank_real = np.argsort(np.argsort(-imp_real))
        rank_synth = np.argsort(np.argsort(-imp_synth))

        corr, _ = spearmanr(rank_real, rank_synth)
        # Spearman is in [-1, 1]; clip at 0 for scoring (negative means inverted)
        return float(max(0.0, corr))

    def _encode_dataframe(
        self, df: pd.DataFrame, reference_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Label-encode non-numeric columns so RF can consume the data.

        When reference_df is provided the encoder is fitted on the union of
        both DataFrames' values, guaranteeing that the same category string
        gets the same integer code in both — required for the discriminator.
        """
        df = df.copy()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                if reference_df is not None and col in reference_df.columns:
                    combined = pd.concat([df[col], reference_df[col]]).astype(str)
                    le.fit(combined)
                else:
                    le.fit(df[col].astype(str))
                df[col] = le.transform(df[col].astype(str))
        return df

    def _print_summary(self, results):
        print("\n=== Consistency Evaluation ===")
        print(f"Overall consistency score: {results['consistency_score']:.4f}")

        acc = results["discriminator_accuracy"]
        flag = "  [RED FLAG - model is easily detectable]" if acc > 0.70 else ""
        print(f"\nDiscriminator accuracy: {acc:.4f}{flag}")
        print(
            f"Discriminator score:    {results['discriminator_score']:.4f}  (target ~1.0, i.e. accuracy ~0.5)"
        )

        if results["constraint_violations"]:
            print("\nConstraint violations:")
            for constraint, rate in results["constraint_violations"].items():
                if constraint != "overall_violation_rate":
                    print(f"  {constraint:<40} {rate * 100:.1f}%")
            print(
                f"  {'Overall violation rate':<40} {results['constraint_violations']['overall_violation_rate'] * 100:.1f}%"
            )
            print(f"  Constraint score: {results['constraint_score']:.4f}")
        else:
            print("\nNo constraints defined — constraint check skipped.")

        if results["feature_importance_spearman"] is not None:
            print(
                f"\nFeature importance Spearman correlation: {results['feature_importance_spearman']:.4f}"
            )
            print("  (1.0 = identical feature rankings, 0.0 = completely different)")
