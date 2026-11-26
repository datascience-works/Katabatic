from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel

from .utils import infer_feature_types, is_integer_valued

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


class NaiveBayesModel(BaseModel):
    """
    Class-conditional Naive Bayes generator (simple generative baseline).

    Assumptions:
    - Features are conditionally independent given the class (target).
    - Categorical features -> multinomial with Laplace smoothing.
    - Continuous features  -> Gaussian per class.

    Original fitting/generation logic by Rishi Goyal (T1 2026), adapted to
    match the current Katabatic Model interface (confirmed against the real
    CTGANModel implementation in katabatic/models/ctgan/models.py).
    """

    ARTIFACT_STATE_FILES = ("naivebayes_state.pkl",)

    def __init__(self, laplace_alpha: float = 1.0, seed: int = 42) -> None:
        super().__init__()
        self.check_dependencies()

        self.laplace_alpha = laplace_alpha
        self.seed = seed
        self.target_col: str | None = None

        self.classes_ = None
        self.class_probs_ = None
        self.features_ = None
        self.feature_types_ = {}
        self._cont_bounds_ = {}
        self._continuous_is_int_ = {}

        self._cat_probs_ = {}
        self._cont_stats_ = {}
        self._n_train_rows: int | None = None

        self._rng = np.random.default_rng(seed)

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas", "sklearn"]

    def _compute_class_distribution(self, df: pd.DataFrame):
        counts = df[self.target_col].value_counts(normalize=True)
        self.classes_ = counts.index.to_numpy()
        self.class_probs_ = counts.to_numpy()

    def _fit_categorical_feature(self, df: pd.DataFrame, feature: str):
        self._cat_probs_.setdefault(feature, {})

        col = df[feature]
        if pd.api.types.is_float_dtype(col) and is_integer_valued(col):
            col = col.astype("Int64")

        all_values = sorted(col.dropna().unique().tolist(), key=str)
        num_values = len(all_values)
        if num_values == 0:
            raise ValueError(f"Categorical column '{feature}' has no values.")

        n_missing = int(col.isna().sum())
        if n_missing:
            logger.warning(
                "Column '%s': %d missing values ignored during fit.", feature, n_missing
            )

        for cls in self.classes_:
            col_c = col[df[self.target_col] == cls]
            counts = col_c.value_counts().reindex(all_values, fill_value=0)
            N = counts.sum()
            alpha = self.laplace_alpha
            probs = (counts + alpha) / (N + alpha * num_values)
            self._cat_probs_[feature][cls] = probs.to_dict()

    def _fit_continuous_feature(self, df: pd.DataFrame, feature: str):
        self._cont_stats_.setdefault(feature, {})

        all_vals = (
            pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy(dtype=float)
        )
        if all_vals.size == 0:
            raise ValueError(f"Continuous column '{feature}' has no numeric values.")
        n_missing = len(df) - all_vals.size
        if n_missing:
            logger.warning(
                "Column '%s': %d missing/non-numeric values ignored during fit.",
                feature,
                n_missing,
            )
        self._cont_bounds_[feature] = (float(all_vals.min()), float(all_vals.max()))

        for cls in self.classes_:
            mask = df[self.target_col] == cls
            vals = (
                pd.to_numeric(df.loc[mask, feature], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )

            if vals.size == 0:
                logger.warning(
                    "Column '%s': no values for class %r, using all rows.", feature, cls
                )
                vals = all_vals

            mean = float(vals.mean())
            std = float(vals.std())
            if not np.isfinite(std) or std == 0.0:
                std = 1e-6

            self._cont_stats_[feature][cls] = (mean, std)

    def _compute_class_counts(self, n_rows: int) -> np.ndarray:
        raw_counts = self.class_probs_ * n_rows
        base_counts = np.floor(raw_counts).astype(int)
        remainder = int(n_rows - base_counts.sum())

        if remainder > 0:
            fractional = raw_counts - base_counts
            top_up = np.argsort(-fractional, kind="stable")[:remainder]
            base_counts[top_up] += 1

        return base_counts

    def _fit(self, df: pd.DataFrame, categorical_cols=None, continuous_cols=None):
        """Fit the class-conditional distributions on a combined real dataset."""
        self.features_ = [c for c in df.columns if c != self.target_col]
        self.feature_types_ = infer_feature_types(
            df, self.features_, categorical_cols, continuous_cols
        )
        self._continuous_is_int_ = {
            c: is_integer_valued(df[c])
            for c, t in self.feature_types_.items()
            if t == "continuous"
        }

        self._compute_class_distribution(df)

        self._cat_probs_.clear()
        self._cont_stats_.clear()
        self._cont_bounds_.clear()

        for feature in self.features_:
            if self.feature_types_[feature] == "categorical":
                self._fit_categorical_feature(df, feature)
            else:
                self._fit_continuous_feature(df, feature)

        self.is_fitted = True
        self._n_train_rows = len(df)
        logger.info(
            "NaiveBayes fitted on %d rows, %d features, %d classes.",
            len(df),
            len(self.features_),
            len(self.classes_),
        )
        return self

    def _generate(self, n_rows: int) -> pd.DataFrame:
        """Generate a synthetic dataset of size n_rows as a DataFrame."""
        if self.classes_ is None:
            raise RuntimeError("Call train() before sampling.")

        class_counts = self._compute_class_counts(n_rows)
        parts = []

        for cls, n_c in zip(self.classes_, class_counts):
            n_c = int(n_c)
            if n_c <= 0:
                continue

            data_cls = {}
            for feature in self.features_:
                if self.feature_types_[feature] == "categorical":
                    probs_dict = self._cat_probs_[feature][cls]
                    values = pd.Series(list(probs_dict.keys())).to_numpy()
                    probs = np.fromiter(probs_dict.values(), dtype=float)
                    probs = probs / probs.sum()
                    idx = self._rng.choice(len(values), size=n_c, p=probs)
                    data_cls[feature] = values[idx]
                else:
                    mean, std = self._cont_stats_[feature][cls]
                    vals = self._rng.normal(loc=mean, scale=std, size=n_c)
                    bounds = self._cont_bounds_.get(feature)
                    if bounds is not None:
                        vals = np.clip(vals, bounds[0], bounds[1])
                    if self._continuous_is_int_.get(feature, False):
                        vals = np.rint(vals).astype(np.int64)
                    data_cls[feature] = vals

            df_cls = pd.DataFrame(data_cls)
            df_cls[self.target_col] = cls
            parts.append(df_cls)

        if not parts:
            raise RuntimeError(
                "No synthetic rows generated. Check fit and class distribution."
            )

        df_synth = pd.concat(parts, ignore_index=True)
        order = self._rng.permutation(len(df_synth))
        logger.debug("Generated %d synthetic rows.", len(df_synth))
        return df_synth.iloc[order].reset_index(drop=True)

    # ----------------- required Model interface (matches CTGANModel exactly) -----------------

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        categorical_cols: list | None = None,
        continuous_cols: list | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> NaiveBayesModel:
        """
        Load data from data_dir (train_full.csv, or x_train.csv + y_train.csv),
        fit the model, generate synthetic data, and save it to synthetic_dir.
        Mirrors CTGANModel.train() exactly so it plugs into the same
        benchmarks/examples/ run scripts and runner.py pipeline.
        """
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. "
                    "Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError(
                    "y_train.csv must have exactly one column (the target)."
                )
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self.target_col = df.columns[-1]
        self._fit(
            df, categorical_cols=categorical_cols, continuous_cols=continuous_cols
        )

        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "naivebayes")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n_samples=len(df))
        x_synth = df_s[df.columns[:-1]].copy()
        y_synth = df_s[[self.target_col]].copy()

        x_synth.to_csv(os.path.join(synth_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(synth_dir, "y_synth.csv"), index=False, header=True)

        logger.info("NaiveBayes synthetic data saved to: %s", synth_dir)
        self._maybe_save_artifact_state(artifact_state_dir)
        return self

    def sample(self, n_samples: int | None = None, *args, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data as a DataFrame with the same column order as
        training data (features..., target last), matching CTGANModel.sample().
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        n_rows = int(n_samples) if n_samples is not None else self._n_train_rows
        df_synth = self._generate(n_rows)
        ordered_cols = self.features_ + [self.target_col]
        return df_synth[ordered_cols]

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        state = {
            "laplace_alpha": self.laplace_alpha,
            "seed": self.seed,
            "target_col": self.target_col,
            "classes_": self.classes_,
            "class_probs_": self.class_probs_,
            "features_": self.features_,
            "feature_types_": self.feature_types_,
            "_continuous_is_int_": self._continuous_is_int_,
            "_cat_probs_": self._cat_probs_,
            "_cont_stats_": self._cont_stats_,
            "_cont_bounds_": self._cont_bounds_,
            "n_train_rows": self._n_train_rows,
        }
        os.makedirs(artifact_state_dir, exist_ok=True)
        state_path = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("NaiveBayes state saved to: %s", state_path)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> NaiveBayesModel:
        state_path = cls._require_state_file(store, ref)
        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(laplace_alpha=state["laplace_alpha"], seed=state["seed"])
        instance.target_col = state["target_col"]
        instance.classes_ = state["classes_"]
        instance.class_probs_ = state["class_probs_"]
        instance.features_ = state["features_"]
        instance.feature_types_ = state["feature_types_"]
        instance._continuous_is_int_ = state["_continuous_is_int_"]
        instance._cat_probs_ = state["_cat_probs_"]
        instance._cont_stats_ = state["_cont_stats_"]
        instance._cont_bounds_ = state.get("_cont_bounds_", {})
        instance._n_train_rows = state.get("n_train_rows")
        instance.is_fitted = True
        return instance
