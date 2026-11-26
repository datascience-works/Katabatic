import sys

import numpy as np
import pandas as pd
import sklearn.base as skbase
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from katabatic.evaluate.base_evaluation import Evaluation


def _make_classifiers():
    """Return a fresh dict of classifier pipelines."""
    return {
        "LR": SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "LinearSVM": SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
                ),
            ]
        ),
        "MLP": SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(100,), max_iter=500, random_state=42
                    ),
                ),
            ]
        ),
    }


class UtilityEvaluation(Evaluation):
    """
    Utility dimension: measures how usable synthetic data is as a substitute
    for real data in downstream machine learning tasks.

    Runs two protocols:
      - TSTR (Train on Synthetic, Test on Real): trains each classifier on
        k folds of synthetic data, evaluates on the held-out test set.
      - TRTR (Train on Real, Test on Real): trains on real CV folds, evaluates
        on the held-out test set. Used as the performance ceiling.

    When test_data is provided both protocols evaluate on the same held-out rows,
    making the TRTR - TSTR delta a clean apples-to-apples comparison. Without
    test_data both fall back to evaluating on folds of real_data.

    Reports accuracy, weighted F1 (``f1``), macro-F1 (``macro_f1``), binary
    AUC when available, and ``per_class_recall`` keyed by original class names.
    Each metric has a mean and population standard deviation across folds.
    Macro-F1 and recall use the same classes across both protocols and all
    folds; undefined class scores are zero.

    The utility score is 1 - average(TRTR - TSTR delta) across accuracy,
    weighted F1, macro-F1 and available AUC, clipped to [0, 1]. Per-class
    recall deltas are diagnostic and do not change the score's weighting.
    A score of 1.0 means synthetic data matches or exceeds real performance
    on average.

    Parameters
    ----------
    real_data : pd.DataFrame
        Training split of the real dataset, including the target column.
    synthetic_data : pd.DataFrame
        Synthetic dataset with the same schema as real_data.
    target_col : str
        Name of the target (label) column.
    test_data : pd.DataFrame, optional
        Held-out test split. When provided, both TSTR and TRTR test on this
        set instead of on folds of real_data.
    n_folds : int
        Number of stratified CV folds (default: 5).
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str,
        test_data: pd.DataFrame = None,
        n_folds: int = 5,
    ):
        super().__init__(real_data, synthetic_data)
        self.target_col = target_col
        self.test_data = test_data
        self.n_folds = n_folds
        self._labels = None
        self._class_names = None

    def evaluate(self) -> dict:
        X_synth = self._encode(
            self.synthetic_data.drop(columns=[self.target_col])
        ).values
        X_real = self._encode(self.real_data.drop(columns=[self.target_col])).values

        y_real_raw = self.real_data[self.target_col]
        y_synth_raw = self.synthetic_data[self.target_col]
        y_test_raw = (
            self.test_data[self.target_col]
            if self.test_data is not None
            else y_real_raw
        )

        if pd.api.types.is_numeric_dtype(y_real_raw):
            y_real = y_real_raw.values
            y_synth = pd.to_numeric(y_synth_raw, errors="coerce").values
            y_test = pd.to_numeric(y_test_raw, errors="coerce").values
        else:
            le = LabelEncoder()
            sources = [y_real_raw, y_synth_raw]
            if self.test_data is not None:
                sources.append(y_test_raw)
            le.fit(pd.concat(sources).astype(str))
            y_real = le.transform(y_real_raw.astype(str))
            y_synth = le.transform(y_synth_raw.astype(str))
            y_test = le.transform(y_test_raw.astype(str))

        self._labels = np.unique(np.concatenate([y_real, y_synth, y_test]))
        if pd.api.types.is_numeric_dtype(y_real_raw):
            real_class_names = {label: str(label) for label in y_real_raw.unique()}
            self._class_names = [
                real_class_names.get(label, str(label)) for label in self._labels
            ]
        else:
            self._class_names = le.inverse_transform(self._labels).tolist()

        if self.test_data is not None:
            X_test = self._encode(self.test_data.drop(columns=[self.target_col])).values
        else:
            X_test = X_real

        is_binary = len(np.unique(y_real)) == 2
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        tstr_results = {}
        trtr_results = {}

        for name, clf in _make_classifiers().items():
            tstr_results[name] = self._tstr(
                clf, X_synth, y_synth, X_test, y_test, cv, is_binary
            )

        for name, clf in _make_classifiers().items():
            trtr_results[name] = self._trtr(
                clf, X_real, y_real, X_test, y_test, cv, is_binary
            )

        delta = self._compute_delta(tstr_results, trtr_results)
        utility_score = self._compute_score(delta)

        results = {
            "tstr": tstr_results,
            "trtr": trtr_results,
            "delta": delta,
            "utility_score": round(utility_score, 4),
        }

        self._print_summary(results)
        return results

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode non-numeric columns so classifiers can consume them."""
        df = df.copy()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                sources = [self.real_data[col], self.synthetic_data[col]]
                if self.test_data is not None and col in self.test_data.columns:
                    sources.append(self.test_data[col])
                le.fit(pd.concat(sources).astype(str))
                df[col] = le.transform(df[col].astype(str))
        return df

    def _tstr(self, clf, X_synth, y_synth, X_test, y_test, cv, is_binary):
        """Train on each synthetic CV fold, test on the held-out test set."""
        fold_metrics = {}
        skipped = 0

        for train_idx, _ in cv.split(X_synth, y_synth):
            if len(np.unique(y_synth[train_idx])) < 2:
                skipped += 1
                continue
            clf_copy = skbase.clone(clf)
            clf_copy.fit(X_synth[train_idx], y_synth[train_idx])
            scores = self._score(clf_copy, X_test, y_test, is_binary)
            self._append_fold_scores(fold_metrics, scores)

        if skipped == self.n_folds:
            raise ValueError(
                "Synthetic data is too class-imbalanced — all CV folds contained "
                "only one class. The model did not learn the class distribution."
            )

        return self._aggregate(fold_metrics)

    def _trtr(self, clf, X_real, y_real, X_test, y_test, cv, is_binary):
        """Train on real CV folds, test on the held-out test set."""
        fold_metrics = {}

        use_held_out = self.test_data is not None
        for train_idx, test_idx in cv.split(X_real, y_real):
            clf_copy = skbase.clone(clf)
            clf_copy.fit(X_real[train_idx], y_real[train_idx])
            # Use held-out test set when available; fall back to the CV fold
            X_eval = X_test if use_held_out else X_real[test_idx]
            y_eval = y_test if use_held_out else y_real[test_idx]
            scores = self._score(clf_copy, X_eval, y_eval, is_binary)
            self._append_fold_scores(fold_metrics, scores)

        return self._aggregate(fold_metrics)

    def _score(self, clf, X, y, is_binary):
        metrics = {}
        y_pred = clf.predict(X)
        labels = self._labels
        if labels is None:
            labels = np.unique(np.concatenate([y, clf.classes_]))
        class_names = self._class_names
        if class_names is None:
            class_names = [str(label) for label in labels]
        metrics["accuracy"] = accuracy_score(y, y_pred)
        metrics["f1"] = f1_score(y, y_pred, average="weighted", zero_division=0)
        metrics["macro_f1"] = f1_score(
            y, y_pred, labels=labels, average="macro", zero_division=0
        )
        recalls = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)
        metrics["per_class_recall"] = dict(zip(class_names, map(float, recalls)))
        if is_binary:
            try:
                y_prob = clf.predict_proba(X)[:, 1]
                metrics["auc"] = roc_auc_score(y, y_prob)
            except Exception as e:
                print(f"  [WARNING] AUC could not be computed: {e}")
        return metrics

    @staticmethod
    def _append_fold_scores(fold_metrics, scores):
        for metric, value in scores.items():
            if metric == "per_class_recall":
                per_class = fold_metrics.setdefault(metric, {})
                for label, recall in value.items():
                    per_class.setdefault(label, []).append(recall)
            else:
                fold_metrics.setdefault(metric, []).append(value)

    def _aggregate(self, fold_metrics):
        return {
            metric: self._aggregate(vals)
            if isinstance(vals, dict)
            else {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
            for metric, vals in fold_metrics.items()
            if vals
        }

    def _compute_delta(self, tstr_results, trtr_results):
        delta = {}
        for clf_name in tstr_results:
            delta[clf_name] = {}
            for metric in tstr_results[clf_name]:
                if metric in trtr_results.get(clf_name, {}):
                    if metric == "per_class_recall":
                        tstr_recalls = tstr_results[clf_name][metric]
                        trtr_recalls = trtr_results[clf_name][metric]
                        delta[clf_name][metric] = {
                            label: round(
                                trtr_recalls[label]["mean"]
                                - tstr_recalls[label]["mean"],
                                4,
                            )
                            for label in tstr_recalls
                            if label in trtr_recalls
                        }
                        continue
                    gap = (
                        trtr_results[clf_name][metric]["mean"]
                        - tstr_results[clf_name][metric]["mean"]
                    )
                    delta[clf_name][metric] = round(gap, 4)
        return delta

    def _compute_score(self, delta):
        score_metrics = {"accuracy", "f1", "macro_f1", "auc"}
        all_deltas = [
            value
            for clf_delta in delta.values()
            for metric, value in clf_delta.items()
            if metric in score_metrics
        ]
        if not all_deltas:
            return 0.0
        avg_delta = float(np.mean(all_deltas))
        return max(0.0, min(1.0, 1.0 - avg_delta))

    def _print_summary(self, results):
        print("\n=== Utility Evaluation ===")
        print(f"Overall utility score: {results['utility_score']:.4f}")
        print(
            f"\n{'Classifier':<12} {'Metric':<24} {'TSTR mean':<12} {'TRTR mean':<12} {'Delta':<8}"
        )
        print("-" * 70)
        for clf in results["tstr"]:
            for metric in results["tstr"][clf]:
                if metric == "per_class_recall":
                    for label, stats in results["tstr"][clf][metric].items():
                        tstr_m = stats["mean"]
                        trtr_m = results["trtr"][clf][metric][label]["mean"]
                        d = results["delta"][clf][metric][label]
                        name = f"recall (class={label})"
                        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                        name = name.encode(encoding, errors="backslashreplace").decode(
                            encoding
                        )
                        print(
                            f"{clf:<12} {name:<24} {tstr_m:<12.4f} {trtr_m:<12.4f} {d}"
                        )
                    continue
                tstr_m = results["tstr"][clf][metric]["mean"]
                trtr_m = results["trtr"][clf][metric]["mean"]
                d = results["delta"][clf].get(metric, "-")
                print(f"{clf:<12} {metric:<24} {tstr_m:<12.4f} {trtr_m:<12.4f} {d}")
