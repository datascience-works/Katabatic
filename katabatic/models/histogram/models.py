from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model

from .utils import get_column_distribution, sample_column

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


class HistogramModel(Model):
    """
    Simple univariate histogram-based tabular data generator.

    For each column, we estimate the empirical distribution (value_counts)
    and then sample from it independently for each synthetic row.
    """

    ARTIFACT_STATE_FILES = ("histogram_state.pkl",)

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.check_dependencies()

        self.random_state = random_state
        self.column_distributions = {}
        self.columns_ = None
        self._n_train_rows: int | None = None
        self._rng = np.random.default_rng(random_state)

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas"]

    def _fit(self, df: pd.DataFrame) -> HistogramModel:
        """Learn per-column empirical distributions from the real data."""
        if df.empty:
            raise ValueError("Cannot train HistogramModel on an empty dataset.")

        self.columns_ = list(df.columns)
        self.column_distributions = {
            col: get_column_distribution(df[col]) for col in self.columns_
        }
        self._n_train_rows = len(df)
        self.is_fitted = True
        return self

    def _generate(self, n_rows: int) -> pd.DataFrame:
        """Generate synthetic data by sampling each column independently."""
        data = {}
        for col in self.columns_:
            values, probs = self.column_distributions[col]
            data[col] = sample_column(self._rng, values, probs, n_rows)

        return pd.DataFrame(data, columns=self.columns_)

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> HistogramModel:
        """
        Load data from data_dir (train_full.csv, or x_train.csv + y_train.csv),
        fit the model, generate synthetic data, and save it to synthetic_dir.
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
            df = pd.concat([X, y], axis=1)

        self._fit(df)

        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "histogram")
        os.makedirs(synth_dir, exist_ok=True)

        target_col = df.columns[-1]
        df_s = self.sample(n_samples=len(df))
        df_s[df.columns[:-1]].to_csv(
            os.path.join(synth_dir, "x_synth.csv"), index=False
        )
        df_s[[target_col]].to_csv(
            os.path.join(synth_dir, "y_synth.csv"), index=False, header=True
        )

        print(f"[Histogram] Synthetic data saved to: {synth_dir}")
        self._maybe_save_artifact_state(artifact_state_dir)
        return self

    def sample(
        self,
        n_samples: int | None = None,
        *args,
        seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the Histogram model.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic rows to generate. Defaults to the number of
            rows the model was trained on.
        seed : int, optional
            Random seed for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset with the same columns as the training data.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        n_rows = int(n_samples) if n_samples is not None else self._n_train_rows
        return self._generate(n_rows)

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        state = {
            "random_state": self.random_state,
            "columns_": self.columns_,
            "column_distributions": self.column_distributions,
            "n_train_rows": self._n_train_rows,
        }
        os.makedirs(artifact_state_dir, exist_ok=True)
        state_path = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> HistogramModel:
        state_path = cls._require_state_file(store, ref)
        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(random_state=state["random_state"])
        instance.columns_ = state["columns_"]
        instance.column_distributions = state["column_distributions"]
        instance._n_train_rows = state["n_train_rows"]
        instance.is_fitted = True
        return instance
