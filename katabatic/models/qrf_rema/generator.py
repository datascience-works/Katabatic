# katabatic/models/qrf/generator.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor


class QRFSequentialGenerator:
    """
    Sequential conditional generator using Quantile Regression Forests.

    Trains one QRF per column:
    P(X_j | X_1, ..., X_{j-1})
    """

    def __init__(self):
        self.columns: list[str] = []
        self.models: dict[str, RandomForestQuantileRegressor] = {}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit one QRF model per column using previous columns as inputs.
        """
        self.columns = list(data.columns)
        self.models = {}

        for i, col in enumerate(self.columns):
            if i == 0:
                # First column: fit unconditional model (dummy input)
                X = np.zeros((len(data), 1))
            else:
                X = data[self.columns[:i]].values

            y = data[col].values

            model = RandomForestQuantileRegressor()
            model.fit(X, y)

            self.models[col] = model

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate n synthetic rows sequentially.
        """
        synthetic = pd.DataFrame(index=range(n))

        for i, col in enumerate(self.columns):
            model = self.models[col]

            if i == 0:
                X = np.zeros((n, 1))
            else:
                X = synthetic[self.columns[:i]].values

            # Sample random quantiles (uniform) → stochastic generation
            q = np.random.uniform(0.05, 0.95, size=n)

            values = model.predict(X, quantiles=q)
            synthetic[col] = values

        return synthetic
