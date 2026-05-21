from __future__ import annotations

import csv
import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from katabatic.evaluate.base_evaluation import Evaluation
from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.ids import new_eval_id
from katabatic.artifacts.refs import DatasetRef, EvaluationRef, ModelRef


def _jsd_kld_from_probs(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = np.asarray(p, dtype=float).ravel() + eps
    q = np.asarray(q, dtype=float).ravel() + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    kld_sym = 0.5 * (kl_pq + kl_qp)
    return jsd, kld_sym


def _marginal_numeric_jsd_kld(
    r: pd.Series, s: pd.Series, n_bins: int
) -> tuple[float, float]:
    r = pd.to_numeric(r, errors="coerce")
    s = pd.to_numeric(s, errors="coerce")
    combo = pd.concat([r.dropna(), s.dropna()])
    if combo.empty:
        return 0.0, 0.0
    edges = np.histogram_bin_edges(combo, bins=min(n_bins, max(3, int(combo.nunique()))))
    pr, _ = np.histogram(r.dropna(), bins=edges)
    ps, _ = np.histogram(s.dropna(), bins=edges)
    pr = pr.astype(float)
    ps = ps.astype(float)
    if pr.sum() == 0 or ps.sum() == 0:
        return 0.0, 0.0
    pr /= pr.sum()
    ps /= ps.sum()
    return _jsd_kld_from_probs(pr, ps)


def _marginal_cat_jsd_kld(r: pd.Series, s: pd.Series) -> tuple[float, float]:
    cats = pd.unique(pd.concat([r, s], ignore_index=True).astype(str))
    pr = r.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).values.astype(float)
    ps = s.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).values.astype(float)
    return _jsd_kld_from_probs(pr, ps)


def _mixed_feature_matrices(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if list(real_x.columns) != list(synth_x.columns):
        raise ValueError("real and synthetic feature columns must match")
    parts_r: list[np.ndarray] = []
    parts_s: list[np.ndarray] = []
    for col in real_x.columns:
        rc = real_x[col]
        sc = synth_x[col]
        if pd.api.types.is_numeric_dtype(rc) and not pd.api.types.is_bool_dtype(rc):
            rr = pd.to_numeric(rc, errors="coerce").astype(float).to_numpy().reshape(-1, 1)
            sr = pd.to_numeric(sc, errors="coerce").astype(float).to_numpy().reshape(-1, 1)
            scaler = StandardScaler()
            parts_r.append(scaler.fit_transform(rr))
            parts_s.append(scaler.transform(sr))
        else:
            dr = pd.get_dummies(rc.astype(str), prefix=str(col))
            ds = pd.get_dummies(sc.astype(str), prefix=str(col))
            dr, ds = dr.align(ds, join="outer", axis=1, fill_value=0)
            parts_r.append(dr.values.astype(float))
            parts_s.append(ds.values.astype(float))
    return np.hstack(parts_r), np.hstack(parts_s)


class StatisticalFidelityEvaluation(Evaluation):
    """JSD / symmetric KLD on univariate marginals and DCR vs real training rows."""

    def __init__(
        self,
        synthetic_dir: str,
        real_train_dir: str,
        *,
        n_bins: int = 10,
        **kwargs,
    ):
        kwargs.pop("real_test_dir", None)
        self.synthetic_dir = synthetic_dir
        self.real_train_dir = real_train_dir
        self.n_bins = int(n_bins)
        self._artifact_store: Optional[ArtifactStore] = kwargs.pop("_artifact_store", None)
        self._evaluation_ref: Optional[EvaluationRef] = kwargs.pop("_evaluation_ref", None)
        self._artifact_report_relpath: Optional[str] = kwargs.pop("_artifact_report_relpath", None)

        self.x_real = pd.read_csv(os.path.join(real_train_dir, "x_train.csv"))
        self.y_real = pd.read_csv(os.path.join(real_train_dir, "y_train.csv")).iloc[:, 0]
        self.x_synth = pd.read_csv(os.path.join(synthetic_dir, "x_synth.csv"))
        self.y_synth = pd.read_csv(os.path.join(synthetic_dir, "y_synth.csv")).iloc[:, 0]

    @classmethod
    def from_artifact(
        cls,
        store: ArtifactStore,
        model_ref: ModelRef,
        dataset_ref: DatasetRef,
        eval_run_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple["StatisticalFidelityEvaluation", EvaluationRef]:
        eval_run_id = eval_run_id or new_eval_id()
        eval_ref = EvaluationRef(
            evaluation_type="fidelity",
            eval_run_id=eval_run_id,
            model_name=model_ref.model_name,
            dataset_name=dataset_ref.dataset_name,
            dataset_version=dataset_ref.dataset_version,
            train_run_id=model_ref.train_run_id,
            test_dataset_version=dataset_ref.dataset_version,
        )
        store.open_path(eval_ref.root_relpath).mkdir(parents=True, exist_ok=True)
        synthetic_dir = str(store.open_path(model_ref.synthetic_relpath))
        real_train_dir = str(store.open_path(dataset_ref.train_relpath))
        skip = frozenset({
            "synthetic_dir",
            "real_train_dir",
            "real_test_dir",
            "_artifact_store",
            "_evaluation_ref",
            "_artifact_report_relpath",
        })
        init_kw = {k: v for k, v in kwargs.items() if k not in skip}
        inst = cls(
            synthetic_dir,
            real_train_dir,
            _artifact_store=store,
            _evaluation_ref=eval_ref,
            _artifact_report_relpath=eval_ref.report_relpath,
            **init_kw,
        )
        return inst, eval_ref

    def evaluate(self) -> dict[str, Any]:
        per_col: dict[str, dict[str, float]] = {}
        jsd_list: list[float] = []
        kld_list: list[float] = []

        label_name = self.y_real.name if hasattr(self.y_real, "name") else "label"
        # Label column
        jsd_y, kld_y = _marginal_cat_jsd_kld(
            self.y_real.astype(str), self.y_synth.astype(str)
        )
        per_col[str(label_name)] = {"jsd": jsd_y, "kld_sym": kld_y}
        jsd_list.append(jsd_y)
        kld_list.append(kld_y)

        for c in self.x_real.columns:
            if c not in self.x_synth.columns:
                raise ValueError(f"synthetic X missing column {c!r}")
            rc = self.x_real[c]
            sc = self.x_synth[c]
            if pd.api.types.is_numeric_dtype(rc) and not pd.api.types.is_bool_dtype(rc):
                jsd_c, kld_c = _marginal_numeric_jsd_kld(rc, sc, self.n_bins)
            else:
                jsd_c, kld_c = _marginal_cat_jsd_kld(rc, sc)
            per_col[str(c)] = {"jsd": float(jsd_c), "kld_sym": float(kld_c)}
            jsd_list.append(float(jsd_c))
            kld_list.append(float(kld_c))

        real_mat, synth_mat = _mixed_feature_matrices(self.x_real, self.x_synth)
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(real_mat)
        dists, _ = nn.kneighbors(synth_mat)
        dists = dists.ravel()
        dcr_mean = float(np.mean(dists))
        dcr_p5 = float(np.percentile(dists, 5))

        results: dict[str, Any] = {
            "mean_jsd": float(np.mean(jsd_list)),
            "mean_kld_sym": float(np.mean(kld_list)),
            "dcr_mean": dcr_mean,
            "dcr_p5": dcr_p5,
            "per_column": per_col,
        }

        if self._artifact_store is not None and self._evaluation_ref is not None:
            self._save_results_artifact(results)
        return results

    def _save_results_artifact(self, results: dict[str, Any]) -> None:
        store = self._artifact_store
        ref = self._evaluation_ref
        assert store is not None and ref is not None

        flat_metrics = {
            "mean_jsd": results["mean_jsd"],
            "mean_kld_sym": results["mean_kld_sym"],
            "dcr_mean": results["dcr_mean"],
            "dcr_p5": results["dcr_p5"],
        }
        serializable = {"summary": flat_metrics, "per_column": results["per_column"]}
        store.save_json(ref.metrics_relpath, serializable)

        lines: list[tuple[str, str, float]] = []
        for k, v in flat_metrics.items():
            lines.append(("summary", k, round(float(v), 6)))
        for col, d in results["per_column"].items():
            for mname, val in d.items():
                lines.append((col, mname, round(float(val), 6)))

        p = store.open_path(ref.report_relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Group", "Metric", "Value"])
            writer.writerows(lines)
        print(f"\nFidelity results saved to: {p}")
