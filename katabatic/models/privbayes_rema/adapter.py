from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .models import PrivBayesModel


@dataclass
class PrivBayesAdapter:
    target_col: str = "target"

    epsilon: Optional[float] = 1.0
    max_parents: int = 2
    n_bins: int = 10
    seed: int = 666
    ordering: str = "greedy_mutual_information"

    model: Optional[PrivBayesModel] = None

    def fit(self, x_train: pd.DataFrame, y_train: Optional[pd.Series] = None, **kwargs) -> None:
        df = x_train.copy()
        if y_train is not None:
            df[self.target_col] = y_train.values

        self.model = PrivBayesModel(
            target_col=self.target_col,
            epsilon=self.epsilon,
            max_parents=self.max_parents,
            n_bins=self.n_bins,
            seed=self.seed,
            ordering=self.ordering,
        )
        self.model.fit(df, **kwargs)

    def sample(self, n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.model is None:
            raise RuntimeError("PrivBayesAdapter not fitted. Call fit() first.")

        x_synth, y_synth = self.model.sample(n_samples)

        # Safety: if model returns a full df (no target present), enforce X-only
        if y_synth is None and isinstance(x_synth, pd.DataFrame) and self.target_col in x_synth.columns:
            y_synth = x_synth[self.target_col].copy()
            x_synth = x_synth.drop(columns=[self.target_col])

        return x_synth, y_synth
