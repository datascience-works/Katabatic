import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from katabatic.evaluate.base_evaluation import Evaluation
from katabatic.utils.column_types import get_column_types


class FidelityEvaluation(Evaluation):
    """
    Fidelity dimension: measures how closely the synthetic data mirrors
    the statistical properties of the real data.

    Metrics
    -------
    - Jensen-Shannon Divergence (JSD) per categorical column.
      JSD = 0 means identical distributions, 1 means completely different.
      Score = 1 - avg_JSD.

    - Wasserstein Distance per continuous column, normalized by the real
      column's range so all columns are on a [0, 1] scale.
      Score = 1 - avg_normalized_WD.

    - Pearson correlation matrix element-wise absolute difference.
      Compares how well inter-column relationships are preserved.
      Score = 1 - avg_abs_diff.

    fidelity_score is the mean of the three component scores.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real dataset (features + target, or features only).
    synthetic_data : pd.DataFrame
        Synthetic dataset with the same schema.
    categorical_cols : list[str], optional
        Columns to treat as categorical. Auto-detected from dtypes if omitted.
    continuous_cols : list[str], optional
        Columns to treat as continuous. Auto-detected from dtypes if omitted.
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_cols: list = None,
        continuous_cols: list = None,
    ):
        super().__init__(real_data, synthetic_data)

        if categorical_cols is None and continuous_cols is None:
            self.categorical_cols, self.continuous_cols = get_column_types(
                self.real_data, exclude_last=False
            )
            print(
                "[WARNING] categorical_cols and continuous_cols were not provided. "
                "Auto-detecting from dtypes — integer-encoded categorical columns will be "
                "misclassified as continuous. Pass column types explicitly for accurate results."
            )
        else:
            self.categorical_cols = categorical_cols or []
            self.continuous_cols = continuous_cols or []

    def evaluate(self) -> dict:
        jsd_results = self._compute_jsd()
        wd_results = self._compute_wasserstein()
        corr_diff = self._compute_correlation_diff()

        # Component scores in [0, 1] — higher is better.
        # Check for actual per-column entries, not just the 'avg' sentinel key,
        # to avoid a spurious 1.0 score when all columns were skipped.
        cat_score = round(1.0 - jsd_results["avg"], 4) if len(jsd_results) > 1 else None
        cont_score = round(1.0 - wd_results["avg"], 4) if len(wd_results) > 1 else None
        corr_score = round(1.0 - corr_diff, 4) if corr_diff is not None else None

        active = [s for s in [cat_score, cont_score, corr_score] if s is not None]
        fidelity_score = round(float(np.mean(active)), 4) if active else 0.0

        results = {
            "categorical_jsd": jsd_results,
            "continuous_wasserstein": wd_results,
            "correlation_diff": round(corr_diff, 4) if corr_diff is not None else None,
            "categorical_score": cat_score,
            "continuous_score": cont_score,
            "correlation_score": corr_score,
            "fidelity_score": fidelity_score,
        }

        self._print_summary(results)
        return results

    def _compute_jsd(self) -> dict:
        if not self.categorical_cols:
            return {}

        scores = {}
        for col in self.categorical_cols:
            if col not in self.synthetic_data.columns:
                continue
            real_dist, synth_dist = self._aligned_distributions(
                self.real_data[col], self.synthetic_data[col]
            )
            # jensenshannon returns the JS distance (sqrt of divergence), range [0, 1]
            score = float(jensenshannon(real_dist, synth_dist))
            scores[col] = round(score, 4)

        scores["avg"] = (
            round(float(np.mean(list(scores.values()))), 4) if scores else 0.0
        )
        return scores

    def _aligned_distributions(self, real_col, synth_col):
        """Return two aligned probability arrays over the union of categories."""
        all_cats = set(real_col.dropna().unique()) | set(synth_col.dropna().unique())
        all_cats = sorted(all_cats, key=str)

        real_counts = real_col.value_counts()
        synth_counts = synth_col.value_counts()

        real_dist = np.array([real_counts.get(c, 0) for c in all_cats], dtype=float)
        synth_dist = np.array([synth_counts.get(c, 0) for c in all_cats], dtype=float)

        # Normalise to probability distributions
        real_dist /= real_dist.sum() if real_dist.sum() > 0 else 1
        synth_dist /= synth_dist.sum() if synth_dist.sum() > 0 else 1

        return real_dist, synth_dist

    def _compute_wasserstein(self) -> dict:
        if not self.continuous_cols:
            return {}

        scores = {}
        for col in self.continuous_cols:
            if col not in self.synthetic_data.columns:
                continue

            real_vals = self.real_data[col].dropna().values.astype(float)
            synth_vals = self.synthetic_data[col].dropna().values.astype(float)

            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue

            col_range = real_vals.max() - real_vals.min()
            if col_range == 0:
                # Constant column — WD is either 0 or the absolute shift
                wd_raw = abs(synth_vals.mean() - real_vals.mean())
                scores[col] = round(min(1.0, wd_raw), 4)
                continue

            # Normalise both arrays to [0, 1] using real data range so the
            # Wasserstein distance is directly comparable across columns
            real_norm = (real_vals - real_vals.min()) / col_range
            synth_norm = (synth_vals - real_vals.min()) / col_range

            wd = float(wasserstein_distance(real_norm, synth_norm))
            scores[col] = round(min(1.0, wd), 4)  # clip at 1 for safety

        scores["avg"] = (
            round(float(np.mean(list(scores.values()))), 4) if scores else 0.0
        )
        return scores

    def _compute_correlation_diff(self):
        num_cols = [c for c in self.continuous_cols if c in self.synthetic_data.columns]
        if len(num_cols) < 2:
            return None

        real_corr = self.real_data[num_cols].corr(method="pearson").values
        synth_corr = self.synthetic_data[num_cols].corr(method="pearson").values

        # Replace NaN (constant columns) with 0 before diffing
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)

        # Element-wise absolute difference, average over upper triangle (excl. diagonal)
        diff_matrix = np.abs(real_corr - synth_corr)
        upper_idx = np.triu_indices_from(diff_matrix, k=1)
        avg_diff = (
            float(np.mean(diff_matrix[upper_idx])) if len(upper_idx[0]) > 0 else 0.0
        )

        # avg_diff is in [0, 2] theoretically; divide by 2 to normalise to [0, 1]
        return round(min(1.0, avg_diff / 2.0), 4)

    def _print_summary(self, results):
        print("\n=== Fidelity Evaluation ===")
        print(f"Overall fidelity score: {results['fidelity_score']:.4f}")

        if results["categorical_jsd"]:
            print(
                f"\nCategorical JSD (lower = better)  ->  score: {results['categorical_score']:.4f}"
            )
            for col, val in results["categorical_jsd"].items():
                if col != "avg":
                    print(f"  {col:<30} JSD = {val:.4f}")
            print(f"  {'avg':<30} JSD = {results['categorical_jsd']['avg']:.4f}")

        if results["continuous_wasserstein"]:
            print(
                f"\nContinuous Wasserstein (normalised, lower = better)  ->  score: {results['continuous_score']:.4f}"
            )
            for col, val in results["continuous_wasserstein"].items():
                if col != "avg":
                    print(f"  {col:<30} WD  = {val:.4f}")
            print(f"  {'avg':<30} WD  = {results['continuous_wasserstein']['avg']:.4f}")

        if results["correlation_diff"] is not None:
            print(
                f"\nCorrelation matrix avg diff: {results['correlation_diff']:.4f}  ->  score: {results['correlation_score']:.4f}"
            )
