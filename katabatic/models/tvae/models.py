"""
TVAEModel — Katabatic integration of TVAE.

Wraps the official, published TVAE implementation from the `ctgan` package
(https://github.com/sdv-dev/CTGAN), by the original CTGAN/TVAE paper authors
(Xu et al., 2019), which itself applies the VAE framework of Kingma &
Welling (2013). See utils.py for the full adaptation rationale.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from katabatic.models.base_model import Model as BaseModel

from .utils import (
    TVAEConfig,
    TVAEState,
    evaluate_tvae,
    sample_tvae,
    train_tvae,
)


class TVAEModel(BaseModel):
    """
    TVAE: a Variational Autoencoder for tabular data (Xu et al., 2019),
    wrapping the official `ctgan` package implementation.

    Interface matches other Katabatic model ports (see TabSyn, TabKDE):
        model = TVAEModel()
        model.train(data_dir, categorical_cols=[...], continuous_cols=[...])
        synthetic_df = model.sample(n)
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        compress_dims: tuple = (128, 128),
        decompress_dims: tuple = (128, 128),
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: int = 2,
        seed: int = 42,
        enable_gpu: bool = False,
    ) -> None:
        super().__init__()
        self.config = TVAEConfig(
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            loss_factor=loss_factor,
            seed=seed,
            enable_gpu=enable_gpu,
        )
        self.state: Optional[TVAEState] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["ctgan", "torch"]

    def train(
        self,
        data_dir: str,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TVAEModel":
        """Fit TVAE on data in `data_dir`, then materialize x_synth.csv /
        y_synth.csv for downstream evaluation, matching the convention used
        by the other model ports in this codebase."""
        self.check_dependencies()

        categorical_cols = categorical_cols or []
        continuous_cols = continuous_cols or []

        self.state = train_tvae(
            data_dir=data_dir,
            cfg=self.config,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
        self.is_fitted = True

        synth_dir = synthetic_dir or kwargs.get("synthetic_dir")
        if not synth_dir or not isinstance(synth_dir, str):
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "tvae")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n_samples=self.state.n_train)

        label_col = self.state.columns[-1]
        feature_cols = [c for c in self.state.columns if c != label_col]

        x_synth = df_s[feature_cols]
        y_synth = df_s[label_col]

        x_path = os.path.join(synth_dir, "x_synth.csv")
        y_path = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path, index=False)
        y_synth.to_csv(y_path, index=False, header=True)
        print(f"[TVAE] Synthetic data saved:\n  X -> {x_path}\n  y -> {y_path}")

        return self

    def evaluate(
        self,
        *,
        data_dir: str,
        split: str = "test",
    ) -> float:
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before evaluate().")
        return evaluate_tvae(self.state, data_dir=data_dir, split=split)

    def sample(
        self,
        n_samples: Optional[int] = None,
        save_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before sample().")

        out = sample_tvae(self.state, n_samples=n_samples)

        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out
