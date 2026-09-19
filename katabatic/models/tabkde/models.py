"""
TabKDEModel — Katabatic integration of TabKDE.

Adapted from the official TabKDE repository (https://github.com/tabkde/tabkde-main),
which extends TabSyn (https://github.com/amazon-science/tabsyn).
Original method/paper: TabKDE authors — copula (empirical-distribution) encoding
combined with a generative model over the resulting latent space.

See utils.py for detailed adaptation notes (what was simplified vs. the
original repo, and why).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# NOTE: adjust this import if your base lives elsewhere (matches TabSyn's pattern)
from katabatic.models.base_model import Model as BaseModel

from .utils import (
    TabKDEConfig,
    TabKDEState,
    evaluate_tabkde,
    sample_tabkde,
    train_tabkde,
)


class TabKDEModel(BaseModel):
    """
    TabKDE: encodes tabular rows into a copula (rank-based) latent space,
    trains a diffusion-style generative model over that space, then decodes
    generated latents back into realistic synthetic rows.

    Interface matches other Katabatic model ports (see TabSyn):
        model = TabKDEModel()
        model.train(data_dir, categorical_cols=[...], continuous_cols=[...])
        synthetic_df = model.sample(n)
    """

    def __init__(
        self,
        *,
        diffusion_epochs: int = 1000,
        diffusion_batch_size: int = 512,
        hidden_dim: int = 256,
        diffusion_steps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 50,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = TabKDEConfig(
            diffusion_epochs=diffusion_epochs,
            diffusion_batch_size=diffusion_batch_size,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            device=device,
        )
        self.state: Optional[TabKDEState] = None

    # ---- Base hooks ---------------------------------------------------------

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn", "scipy"]

    def train(
        self,
        data_dir: str,
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TabKDEModel":
        """Fit the copula encoding + diffusion model on data in `data_dir`,
        then materialize x_synth.csv / y_synth.csv for downstream evaluation
        (TSTR / SyntheticEvaluationPipeline), matching the convention used by
        the other model ports in this codebase (e.g. TabSyn)."""
        self.check_dependencies()

        categorical_cols = categorical_cols or []
        continuous_cols = continuous_cols or []

        self.state = train_tabkde(
            data_dir=data_dir,
            cfg=self.config,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
        self.is_fitted = True

        # Decide where to save synthetic data
        synth_dir = synthetic_dir or kwargs.get("synthetic_dir")
        if not synth_dir or not isinstance(synth_dir, str):
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "tabkde")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n_samples=self.state.n_train)

        # Split into X / y for TSTR — label is the last original column
        label_col = self.state.columns[-1]
        feature_cols = [c for c in self.state.columns if c != label_col]

        x_synth = df_s[feature_cols]
        y_synth = df_s[label_col]

        x_path = os.path.join(synth_dir, "x_synth.csv")
        y_path = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path, index=False)
        y_synth.to_csv(y_path, index=False, header=True)
        print(f"[TabKDE] Synthetic data saved:\n  X -> {x_path}\n  y -> {y_path}")

        return self

    def evaluate(
        self,
        *,
        data_dir: str,
        split: str = "test",
    ) -> float:
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before evaluate().")
        return evaluate_tabkde(self.state, data_dir=data_dir, split=split)

    def sample(
        self,
        n_samples: Optional[int] = None,
        save_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic rows. Accepts a plain int (positional) to match
        the calling convention used by benchmarks/examples runner scripts,
        e.g. `model.sample(len(train_df))`."""
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before sample().")

        out = sample_tabkde(self.state, n_samples=n_samples)

        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out
