import csv
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # optional dependency (see pyproject.toml)
    XGBClassifier = None  # type: ignore[misc, assignment]

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.ids import new_eval_id
from katabatic.artifacts.refs import DatasetRef, EvaluationRef, ModelRef
from katabatic.evaluate.base_evaluation import Evaluation


def load_data(synthetic_dir, real_test_dir):
    x_synth = pd.read_csv(os.path.join(synthetic_dir, "x_synth.csv"))
    y_synth = pd.read_csv(os.path.join(
        synthetic_dir, "y_synth.csv")).values.ravel()
    x_test = pd.read_csv(os.path.join(real_test_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(
        real_test_dir, "y_test.csv")).values.ravel()
    return x_synth, y_synth, x_test, y_test


class TSTREvaluation(Evaluation):
    def __init__(self, synthetic_dir, real_test_dir, **kwargs):
        kwargs.pop("real_train_dir", None)
        self.synthetic_dir = synthetic_dir
        self.real_test_dir = real_test_dir
        self._artifact_store: ArtifactStore | None = kwargs.pop("_artifact_store", None)
        self._evaluation_ref: EvaluationRef | None = kwargs.pop("_evaluation_ref", None)
        self._artifact_report_relpath: str | None = kwargs.pop("_artifact_report_relpath", None)

        self.x_train, self.y_train, self.x_test, self.y_test = load_data(
            synthetic_dir, real_test_dir)

    @classmethod
    def from_artifact(
        cls,
        store: ArtifactStore,
        model_ref: ModelRef,
        dataset_ref: DatasetRef,
        eval_run_id: str | None = None,
        **kwargs,
    ) -> tuple["TSTREvaluation", EvaluationRef]:
        eval_run_id = eval_run_id or new_eval_id()
        eval_ref = EvaluationRef(
            evaluation_type="tstr",
            eval_run_id=eval_run_id,
            model_name=model_ref.model_name,
            dataset_name=dataset_ref.dataset_name,
            dataset_version=dataset_ref.dataset_version,
            train_run_id=model_ref.train_run_id,
            test_dataset_version=dataset_ref.dataset_version,
        )
        store.open_path(eval_ref.root_relpath).mkdir(parents=True, exist_ok=True)
        synthetic_dir = str(store.open_path(model_ref.synthetic_relpath))
        real_test_dir = str(store.open_path(dataset_ref.test_relpath))
        report_rel = eval_ref.report_relpath
        skip = frozenset({
            "_artifact_store",
            "_evaluation_ref",
            "_artifact_report_relpath",
            "synthetic_dir",
            "real_test_dir",
            "real_train_dir",
        })
        init_kw = {k: v for k, v in kwargs.items() if k not in skip}
        inst = cls(
            synthetic_dir,
            real_test_dir,
            _artifact_store=store,
            _evaluation_ref=eval_ref,
            _artifact_report_relpath=report_rel,
            **init_kw,
        )
        return inst, eval_ref

    def evaluate(self):
        results = {}
        # Calculate class imbalance ratio for XGBoost
        num_neg = np.sum(self.y_train == 0)
        num_pos = np.sum(self.y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        models: dict[str, Any] = {
            "LR": LogisticRegression(),
            "MLP": MLPClassifier(),
            "RF": RandomForestClassifier(),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(scale_pos_weight=scale_pos_weight)
        else:
            warnings.warn(
                "xgboost is not installed; TSTR will skip the XGBoost classifier. "
                "Install with: pip install xgboost",
                stacklevel=2,
            )
        for name, model in models.items():
            if name in ["LR", "MLP"]:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(self.x_train)
                x_test_scaled = scaler.transform(self.x_test)
                model.fit(x_train_scaled, self.y_train)
                y_pred = model.predict(x_test_scaled)
                y_prob = model.predict_proba(x_test_scaled)[:, 1]
            else:
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                y_prob = model.predict_proba(self.x_test)[:, 1]

            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred, average='weighted')
            }

            # Add AUC for binary classification
            if len(np.unique(self.y_test)) == 2:
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)

            results[name] = metrics

        if self._artifact_store is not None and self._evaluation_ref is not None and self._artifact_report_relpath is not None:
            self._save_results_artifact(results)
        else:
            self.save_results_to_csv(results, self.synthetic_dir)

        print("\nTSTR Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        return results

    def _save_results_artifact(self, results: dict[str, dict[str, float]]) -> None:
        store = self._artifact_store
        ref = self._evaluation_ref
        assert store is not None and ref is not None

        serializable: dict[str, Any] = {
            k: {m: float(v) for m, v in d.items()}
            for k, d in results.items()
        }
        store.save_json(ref.metrics_relpath, serializable)

        report_path = self._artifact_report_relpath
        lines = []
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                lines.append((model_name, metric_name, round(float(value), 4)))

        p = store.open_path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Metric", "Value"])
            writer.writerows(lines)
        print(f"\nResults saved to: {p}")

    @staticmethod
    def save_results_to_csv(results, synthetic_dir):
        parts = os.path.normpath(synthetic_dir).split(os.sep)
        model_name = parts[-1]
        dataset_name = parts[-2]

        results_dir = os.path.join("Results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"{model_name}_tstr.csv")

        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Metric", "Value"])
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    writer.writerow([model_name, metric_name, round(value, 4)])

        print(f"\nResults saved to: {output_path}")
