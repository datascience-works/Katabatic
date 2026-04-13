import os
import csv

import numpy as np
import pandas as pd
import sklearn.base as skbase
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from katabatic.evaluate.base_evaluation import Evaluation


def _make_classifiers():
    """Return a fresh dict of classifier pipelines."""
    return {
        'LR': SklearnPipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'LinearSVM': SklearnPipeline([
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))),
        ]),
        'MLP': SklearnPipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
        ]),
    }


class UtilityEvaluation(Evaluation):
    """
    Utility dimension: measures how usable synthetic data is as a substitute
    for real data in downstream machine learning tasks.

    Runs two protocols:
      - TSTR (Train on Synthetic, Test on Real): trains each classifier on
        k folds of synthetic data, evaluates on the full real dataset each time.
      - TRTR (Train on Real, Test on Real): standard k-fold CV on real data,
        used as the performance ceiling.

    The utility score is 1 - average(TRTR - TSTR delta). A score of 1.0
    means synthetic data performs identically to real data.

    Parameters
    ----------
    real_data : pd.DataFrame
        Full real dataset including the target column.
    synthetic_data : pd.DataFrame
        Synthetic dataset with the same schema as real_data.
    target_col : str
        Name of the target (label) column.
    n_folds : int
        Number of stratified CV folds (default: 5).
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str,
        n_folds: int = 5,
        **kwargs,
    ):
        super().__init__(real_data, synthetic_data, **kwargs)
        self.target_col = target_col
        self.n_folds = n_folds

    def evaluate(self) -> dict:
        # If the data went through our encode_preprocess this is still fine, its just a safe measure in case the users feeds raw data
        X_synth = self._encode(self.synthetic_data.drop(columns=[self.target_col])).values
        X_real = self._encode(self.real_data.drop(columns=[self.target_col])).values

        # Cast targets to the same numeric dtype to avoid type mismatches when
        # models (e.g. CTGAN) return the target column as strings
        y_real  = pd.to_numeric(self.real_data[self.target_col],  errors='coerce').values
        y_synth = pd.to_numeric(self.synthetic_data[self.target_col], errors='coerce').values

        is_binary = len(np.unique(y_real)) == 2
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        tstr_results = {}
        trtr_results = {}

        for name, clf in _make_classifiers().items():
            tstr_results[name] = self._tstr(clf, X_synth, y_synth, X_real, y_real, cv, is_binary)

        for name, clf in _make_classifiers().items():
            trtr_results[name] = self._trtr(clf, X_real, y_real, cv, is_binary)

        delta = self._compute_delta(tstr_results, trtr_results)
        utility_score = self._compute_score(delta)

        results = {
            'tstr': tstr_results,
            'trtr': trtr_results,
            'delta': delta,
            'utility_score': round(utility_score, 4),
        }

        self._print_summary(results)
        return results


    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode non-numeric columns so classifiers can consume them."""
        df = df.copy()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                combined = pd.concat([
                    self.real_data[col], self.synthetic_data[col]
                ]).astype(str)
                le.fit(combined)
                df[col] = le.transform(df[col].astype(str))
        return df

    def _tstr(self, clf, X_synth, y_synth, X_real, y_real, cv, is_binary):
        """Train on each synthetic CV fold, test on ALL real data."""
        fold_metrics = {'accuracy': [], 'f1': [], 'auc': []}
        skipped = 0

        for train_idx, _ in cv.split(X_synth, y_synth):
            if len(np.unique(y_synth[train_idx])) < 2:
                skipped += 1
                continue
            clf_copy = skbase.clone(clf)
            clf_copy.fit(X_synth[train_idx], y_synth[train_idx])
            scores = self._score(clf_copy, X_real, y_real, is_binary)
            for k, v in scores.items():
                fold_metrics[k].append(v)

        if skipped == self.n_folds:
            raise ValueError(
                "Synthetic data is too class-imbalanced — all CV folds contained "
                "only one class. The model did not learn the class distribution."
            )

        return self._aggregate(fold_metrics)

    def _trtr(self, clf, X_real, y_real, cv, is_binary):
        """Standard k-fold CV entirely on real data."""
        fold_metrics = {'accuracy': [], 'f1': [], 'auc': []}

        for train_idx, test_idx in cv.split(X_real, y_real):
            clf_copy = skbase.clone(clf)
            clf_copy.fit(X_real[train_idx], y_real[train_idx])
            scores = self._score(clf_copy, X_real[test_idx], y_real[test_idx], is_binary)
            for k, v in scores.items():
                fold_metrics[k].append(v)

        return self._aggregate(fold_metrics)

    def _score(self, clf, X, y, is_binary):
        metrics = {}
        y_pred = clf.predict(X)
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        if is_binary:
            try:
                y_prob = clf.predict_proba(X)[:, 1]
                metrics['auc'] = roc_auc_score(y, y_prob)
            except Exception:
                pass
        return metrics

    def _aggregate(self, fold_metrics):
        return {
            metric: {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
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
                    gap = (
                        trtr_results[clf_name][metric]['mean']
                        - tstr_results[clf_name][metric]['mean']
                    )
                    delta[clf_name][metric] = round(gap, 4)
        return delta

    def _compute_score(self, delta):
        all_deltas = [v for clf_d in delta.values() for v in clf_d.values()]
        if not all_deltas:
            return 0.0
        avg_delta = float(np.mean(all_deltas))
        return max(0.0, min(1.0, 1.0 - avg_delta))

    def _print_summary(self, results):
        print("\n=== Utility Evaluation ===")
        print(f"Overall utility score: {results['utility_score']:.4f}")
        print(f"\n{'Classifier':<12} {'Metric':<10} {'TSTR mean':<12} {'TRTR mean':<12} {'Delta':<8}")
        print("-" * 56)
        for clf in results['tstr']:
            for metric in results['tstr'][clf]:
                tstr_m = results['tstr'][clf][metric]['mean']
                trtr_m = results['trtr'][clf][metric]['mean']
                d = results['delta'][clf].get(metric, '-')
                print(f"{clf:<12} {metric:<10} {tstr_m:<12.4f} {trtr_m:<12.4f} {d}")

    def save_results(self, results: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'utility_evaluation.csv')

        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mode', 'Classifier', 'Metric', 'Mean', 'Std'])
            for mode, key in [('TSTR', 'tstr'), ('TRTR', 'trtr')]:
                for clf, metrics in results[key].items():
                    for metric, vals in metrics.items():
                        writer.writerow([mode, clf, metric, vals['mean'], vals['std']])
            writer.writerow([])
            writer.writerow(['Delta (TRTR - TSTR)'])
            writer.writerow(['Classifier', 'Metric', 'Delta'])
            for clf, metrics in results['delta'].items():
                for metric, val in metrics.items():
                    writer.writerow([clf, metric, val])
            writer.writerow([])
            writer.writerow(['utility_score', results['utility_score']])

        print(f"Utility results saved to: {path}")
