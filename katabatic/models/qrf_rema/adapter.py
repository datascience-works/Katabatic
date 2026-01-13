# katabatic/models/qrf/adapter.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor
from .generator import QRFSequentialGenerator


class QRFAdapter:
    """
    Quantile Regression Forest adapter for Katabatic.

    Uses the official `quantile-forest` implementation with
    default parameters only (no tuning).
    """

    def __init__(self, target_col: str):
        self.target_col = target_col
        self.generator: QRFSequentialGenerator | None = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        Fit sequential QRF models column-by-column.
        """
        data = x_train.copy()
        data[self.target_col] = y_train.values

        self.generator = QRFSequentialGenerator()
        self.generator.fit(data)

    def sample(self, num_rows: int):
        """
        Generate synthetic samples.
        """
        if self.generator is None:
            raise RuntimeError("QRFAdapter not fitted. Call fit() first.")

        synthetic = self.generator.sample(num_rows)

        x_synth = synthetic.drop(columns=[self.target_col])
        y_synth = synthetic[self.target_col]

        return x_synth, y_synth
