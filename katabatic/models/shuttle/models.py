"""
models.py
---------
Model definitions and training/evaluation helpers for classifying the
Statlog (Shuttle) dataset's 7 operating-state classes.

Two models are provided out of the box:

- RandomForestModel  : strong baseline, handles imbalance reasonably,
                        no scaling required.
- LogisticRegressionModel : simple linear baseline, benefits from the
                        scaled features produced by utils.prepare_dataset.

Both share a common `BaseModel` interface (fit / predict / evaluate /
save / load) so they're interchangeable.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .utils import Dataset


class BaseModel(ABC):
    """Common interface for all models in this package."""

    def __init__(self, **kwargs: Any) -> None:
        self.params: dict[str, Any] = kwargs
        self.estimator = self._build_estimator(**kwargs)
        self.is_fitted: bool = False

    @abstractmethod
    def _build_estimator(self, **kwargs: Any) -> Any:
        """Construct and return the underlying sklearn estimator."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                f"{type(self.estimator).__name__} does not support predict_proba."
            )
        return self.estimator.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Return accuracy, macro-F1, a classification report, and a confusion matrix."""
        self._check_fitted()
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"),
            "classification_report": classification_report(y, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y, preds),
        }

    def save(self, path: str) -> None:
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> BaseModel:
        with open(path, "rb") as f:
            model = pickle.load(f)  # nosec B301: only load trusted model artifacts
        if not isinstance(model, BaseModel):
            raise TypeError(f"Object at {path} is not a BaseModel instance.")
        return model

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} has not been fit yet. Call .fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({status}, params={self.params})"


class RandomForestModel(BaseModel):
    """Random forest classifier. Good default: robust to unscaled features
    and moderately imbalanced classes."""

    def _build_estimator(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        **kwargs: Any,
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )


class LogisticRegressionModel(BaseModel):
    """Multinomial logistic regression baseline. Expects scaled features
    (see utils.prepare_dataset(scale=True))."""

    def _build_estimator(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        **kwargs: Any,
    ) -> LogisticRegression:
        return LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs,
        )


def train_model(model: BaseModel, dataset: Dataset) -> BaseModel:
    """Fit `model` on dataset.X_train / dataset.y_train."""
    return model.fit(dataset.X_train, dataset.y_train)


def evaluate_model(model: BaseModel, dataset: Dataset) -> dict[str, Any]:
    """Evaluate `model` on dataset.X_test / dataset.y_test."""
    return model.evaluate(dataset.X_test, dataset.y_test)
