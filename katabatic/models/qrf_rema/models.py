# katabatic/models/qrf/models.py

from __future__ import annotations

import pandas as pd
from quantile_forest import (
    RandomForestQuantileRegressor,
    ExtraTreesQuantileRegressor,
)


class QRFModel:
    """
    Quantile Regression Forest model wrapper.

    This class wraps the official `quantile-forest` implementation
    and exposes a clean interface for Katabatic.
    """

    def __init__(self, model_type: str = "rf"):
        """
        Parameters
        ----------
        model_type : {"rf", "extra"}
            - "rf"    → RandomForestQuantileRegressor
            - "extra" → ExtraTreesQuantileRegressor
        """
        if model_type == "rf":
            self.model = RandomForestQuantileRegressor()
        elif model_type == "extra":
            self.model = ExtraTreesQuantileRegressor()
        else:
            raise ValueError("model_type must be 'rf' or 'extra'")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the QRF model (no tuning, default parameters only).
        """
        self.model.fit(X.values, y.values)

    def predict(self, X: pd.DataFrame, quantiles=0.5):
        """
        Predict quantiles.
        """
        return self.model.predict(X.values, quantiles=quantiles)
