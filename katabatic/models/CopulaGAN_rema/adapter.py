

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .models import CopulaGANModel


@dataclass
class CopulaGANAdapter:
    """
    Katabatic-style adapter wrapper for CopulaGAN.

    Expected API (matches Katabatic pattern):
    - fit(x_train, y_train)
    - sample(n) -> (x_synth, y_synth)
    """
    target_col: str = "target"
    model: Optional[CopulaGANModel] = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        # Keep kwargs so the pipeline can pass config if needed,
        # but you can run with SDV defaults by passing nothing.
        self.model = CopulaGANModel(target_col=self.target_col, **kwargs)
        self.model.fit(x_train, y_train)

    def sample(self, num_rows: int) -> tuple[pd.DataFrame, pd.Series]:
        if self.model is None:
            raise RuntimeError("CopulaGANAdapter not fitted. Call fit() first.")
        return self.model.sample(num_rows)
