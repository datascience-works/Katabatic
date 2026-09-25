from __future__ import annotations

import json
import os
import pickle

import pandas as pd

from katabatic.artifacts import ArtifactStore, ModelRef
from katabatic.models.base_model import Model as BaseModel

from .utils import KDEModel


class KDESynthesizer(BaseModel):
    """
    Class-conditional KDE / histogram synthesizer for tabular data.

    Continuous features are modeled with a 1D Gaussian KDE per class;
    categorical features with class-conditional histograms; the class
    distribution itself is matched to the real data. No training loop,
    no GPU — fits in closed form via ``sklearn.neighbors.KernelDensity``.

    Ported from the katabatic-mentorship registry (Rishi_Goyal branch)
    and adapted to Katabatic's ``Model`` interface and artifact I/O
    conventions.
    """

    ARTIFACT_STATE_FILES = ("kde_model.pkl",)

    def __init__(
        self,
        *,
        kernel: str = "gaussian",
        bandwidth: float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = {"kernel": kernel, "bandwidth": bandwidth, "seed": seed}
        self._kde: KDEModel | None = None
        self._target_col: str | None = None
        self._feature_cols: list[str] = []
        self._n_train_rows: int | None = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn"]

    def fit(
        self,
        df: pd.DataFrame,
        *,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
    ) -> KDESynthesizer:
        """Fit KDE on a combined feature/target dataframe."""
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        if df.shape[1] < 2:
            raise ValueError(
                "KDE requires at least one feature column and one target column."
            )

        self._target_col = str(df.columns[-1])
        self._feature_cols = [c for c in df.columns if c != self._target_col]

        if categorical_cols is not None:
            resolved_categorical_cols = {
                c for c in categorical_cols if c in self._feature_cols
            }
        else:
            resolved_categorical_cols = None

        self._kde = KDEModel(
            target_col=self._target_col,
            categorical_cols=resolved_categorical_cols,
            kernel=self.cfg["kernel"],
            bandwidth=self.cfg["bandwidth"],
            random_state=self.cfg["seed"],
        )
        self._kde.fit(df)
        self._n_train_rows = len(df)
        self.is_fitted = True

        return self

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
        **kwargs,
    ) -> KDESynthesizer:
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

            df = pd.concat([X, y], axis=1)

        self._target_col = str(df.columns[-1])
        self._feature_cols = [c for c in df.columns if c != self._target_col]

        if categorical_cols is not None:
            resolved_categorical_cols = {
                c for c in categorical_cols if c in self._feature_cols
            }
        else:
            resolved_categorical_cols = self._load_categorical_cols(data_dir)

        self.fit(
            df,
            categorical_cols=(
                sorted(resolved_categorical_cols)
                if resolved_categorical_cols is not None
                else None
            ),
            continuous_cols=continuous_cols,
        )

        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "kde")

        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n_samples=len(df))
        x_synth = df_s[self._feature_cols].copy()
        y_synth = df_s[[self._target_col]].copy()

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")

        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        resolved_categorical_cols = (
            sorted(self._kde.categorical_cols)
            if self._kde is not None and self._kde.categorical_cols
            else []
        )

        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": self._target_col,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "categorical_columns": resolved_categorical_cols,
            },
            "training": self.cfg,
        }

        with open(
            os.path.join(synth_dir, "metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=2)

        print(f"[KDE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")

        self._maybe_save_artifact_state(artifact_state_dir)

        return self

    def _load_categorical_cols(self, data_dir: str) -> set[str] | None:
        """
        Prefer Katabatic's own ``info.json`` (``cat_col_idx``) when present.
        Pipeline data is often already integer-encoded, so dtype alone can't
        tell categorical codes apart from real continuous values.
        """
        info_path = os.path.join(data_dir, "info.json")
        if not os.path.exists(info_path):
            return None

        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        cat_idx = info.get("cat_col_idx")
        if cat_idx is None:
            return None

        return {self._feature_cols[i] for i in cat_idx if i < len(self._feature_cols)}

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        raise NotImplementedError(
            "KDESynthesizer.evaluate() has no meaningful standalone metric "
            "to offer. Use TSTREvaluation for cross-model metrics instead."
        )

    def sample(
        self,
        n_samples: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self._kde is None:
            raise RuntimeError("Call train() before sample().")

        n_rows = int(n_samples) if n_samples is not None else self._n_train_rows
        return self._kde.generate(n_rows)

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        os.makedirs(artifact_state_dir, exist_ok=True)

        state = {
            "cfg": self.cfg,
            "kde": self._kde,
            "target_col": self._target_col,
            "feature_cols": self._feature_cols,
            "n_train_rows": self._n_train_rows,
            "is_fitted": self.is_fitted,
        }

        target = os.path.join(
            artifact_state_dir,
            self.ARTIFACT_STATE_FILES[0],
        )

        with open(target, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(
        cls,
        store: ArtifactStore,
        ref: ModelRef,
    ) -> KDESynthesizer:
        state_path = cls._require_state_file(store, ref)

        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(**state["cfg"])
        instance._kde = state["kde"]
        instance._target_col = state["target_col"]
        instance._feature_cols = state["feature_cols"]
        instance._n_train_rows = state["n_train_rows"]
        instance.is_fitted = state["is_fitted"]

        return instance
