import csv
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.ids import new_eval_id
from katabatic.artifacts.refs import DatasetRef, EvaluationRef, ModelRef
from katabatic.evaluate.base_evaluation import Evaluation
from katabatic.utils.column_types import get_column_types


def _load_fidelity_data(
    synthetic_dir: str, real_test_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct combined (features + target) real/synthetic frames from the
    artifact pipeline's per-column CSVs, mirroring TSTREvaluation's load_data()."""
    x_synth = pd.read_csv(f"{synthetic_dir}/x_synth.csv")
    y_synth = pd.read_csv(f"{synthetic_dir}/y_synth.csv")
    x_test = pd.read_csv(f"{real_test_dir}/x_test.csv")
    y_test = pd.read_csv(f"{real_test_dir}/y_test.csv")
    real_data = pd.concat(
        [x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    synthetic_data = pd.concat(
        [x_synth.reset_index(drop=True), y_synth.reset_index(drop=True)], axis=1
    )
    return real_data, synthetic_data


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
        **kwargs,
    ):
        super().__init__(real_data, synthetic_data)

        # Set only via from_artifact(); when present, evaluate() writes its
        # results into the artifact store instead of just printing/returning them.
        self._artifact_store: ArtifactStore | None = kwargs.pop("_artifact_store", None)
        self._evaluation_ref: EvaluationRef | None = kwargs.pop("_evaluation_ref", None)
        self._artifact_report_relpath: str | None = kwargs.pop(
            "_artifact_report_relpath", None
        )

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

    @classmethod
    def from_artifact(
        cls,
        store: ArtifactStore,
        model_ref: ModelRef,
        dataset_ref: DatasetRef,
        eval_run_id: str | None = None,
        **kwargs,
    ) -> tuple["FidelityEvaluation", EvaluationRef]:
        """Artifact-pipeline adapter: reads the same x/y CSVs TSTREvaluation does,
        recombines them into (real_data, synthetic_data) frames, and wires up
        artifact-mode report writing. Lets FidelityEvaluation run through
        TrainTestSplitPipeline despite its DataFrame-based constructor."""
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
        # Fetch inputs another machine may have written (no-op for local stores).
        store.pull(model_ref.synthetic_relpath)
        store.pull(dataset_ref.test_relpath)
        synthetic_dir = str(store.open_path(model_ref.synthetic_relpath))
        real_test_dir = str(store.open_path(dataset_ref.test_relpath))
        real_data, synthetic_data = _load_fidelity_data(synthetic_dir, real_test_dir)

        skip = frozenset(
            {
                "_artifact_store",
                "_evaluation_ref",
                "_artifact_report_relpath",
                "synthetic_dir",
                "real_test_dir",
                "real_train_dir",
            }
        )
        init_kw = {k: v for k, v in kwargs.items() if k not in skip}
        inst = cls(
            real_data,
            synthetic_data,
            _artifact_store=store,
            _evaluation_ref=eval_ref,
            _artifact_report_relpath=eval_ref.report_relpath,
            **init_kw,
        )
        return inst, eval_ref

    def evaluate(self) -> dict:
        jsd_results = self._compute_jsd()
        wd_results = self._compute_wasserstein()
        corr_diff = self._compute_correlation_diff()
        dcr_mean = self._compute_dcr()

        # Component scores in [0, 1] — higher is better.
        # Check for actual per-column entries, not just the 'avg' sentinel key,
        # to avoid a spurious 1.0 score when all columns were skipped.
        cat_score = round(1.0 - jsd_results["avg"], 4) if len(jsd_results) > 1 else None
        cont_score = round(1.0 - wd_results["avg"], 4) if len(wd_results) > 1 else None
        corr_score = round(1.0 - corr_diff, 4) if corr_diff is not None else None

        active = [s for s in [cat_score, cont_score, corr_score] if s is not None]
        fidelity_score = round(float(np.mean(active)), 4) if active else 0.0
        mean_jsd = jsd_results["avg"] if len(jsd_results) > 1 else None

        results = {
            "categorical_jsd": jsd_results,
            "continuous_wasserstein": wd_results,
            "correlation_diff": round(corr_diff, 4) if corr_diff is not None else None,
            "categorical_score": cat_score,
            "continuous_score": cont_score,
            "correlation_score": corr_score,
            "fidelity_score": fidelity_score,
            "dcr_mean": dcr_mean,
            "summary": {
                "fidelity_score": fidelity_score,
                "mean_jsd": mean_jsd,
                "dcr_mean": dcr_mean,
            },
        }

        self._print_summary(results)

        if (
            self._artifact_store is not None
            and self._evaluation_ref is not None
            and self._artifact_report_relpath is not None
        ):
            self._save_results_artifact(results)

        return results

    def _compute_dcr(self) -> float | None:
        """Distance to Closest Record: for each synthetic row, the Euclidean
        distance (in a jointly-scaled feature space) to its nearest real row,
        averaged. Categorical columns are one-hot encoded on the union of real +
        synthetic categories; continuous columns are min-max scaled by the real
        data's range. Lower means synthetic rows sit closer to real ones — a
        fidelity/privacy-adjacent signal, not a substitute for a formal privacy
        audit. Returns None if there are no usable columns or rows.
        """
        cols = [
            c
            for c in (self.categorical_cols + self.continuous_cols)
            if c in self.real_data.columns and c in self.synthetic_data.columns
        ]
        if not cols:
            return None

        real = self.real_data[cols].dropna()
        synth = self.synthetic_data[cols].dropna()
        if real.empty or synth.empty:
            return None

        real_parts = []
        synth_parts = []

        cat_cols = [c for c in self.categorical_cols if c in cols]
        if cat_cols:
            combined = pd.concat(
                [real[cat_cols].astype(str), synth[cat_cols].astype(str)],
                keys=["real", "synth"],
            )
            encoded = pd.get_dummies(combined, columns=cat_cols)
            real_parts.append(encoded.loc["real"].to_numpy(dtype=float))
            synth_parts.append(encoded.loc["synth"].to_numpy(dtype=float))

        cont_cols = [c for c in self.continuous_cols if c in cols]
        if cont_cols:
            real_cont = real[cont_cols].to_numpy(dtype=float)
            synth_cont = synth[cont_cols].to_numpy(dtype=float)
            col_min = real_cont.min(axis=0)
            col_range = real_cont.max(axis=0) - col_min
            col_range[col_range == 0] = 1.0
            real_parts.append((real_cont - col_min) / col_range)
            synth_parts.append((synth_cont - col_min) / col_range)

        real_matrix = np.concatenate(real_parts, axis=1)
        synth_matrix = np.concatenate(synth_parts, axis=1)

        neighbors = NearestNeighbors(n_neighbors=1).fit(real_matrix)
        distances, _ = neighbors.kneighbors(synth_matrix)
        return round(float(np.mean(distances)), 4)

    def _save_results_artifact(self, results: dict[str, Any]) -> None:
        store = self._artifact_store
        ref = self._evaluation_ref
        assert store is not None and ref is not None

        store.save_json(ref.metrics_relpath, results)

        report_path = self._artifact_report_relpath
        p = store.open_path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for metric, value in results["summary"].items():
                writer.writerow([metric, value])
        store.sync(report_path)
        print(f"\nResults saved to: {p}")

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
