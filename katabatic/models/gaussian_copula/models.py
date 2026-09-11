from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from katabatic.models.base_model import Model

from .utils import load_training_dataframe, resolve_synthetic_dir, save_synthetic_split


class GaussianCopulaModel(Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.metadata = None
        self.train_data = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "GaussianCopulaModel":

        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        data = load_training_dataframe(data_dir)
        self.train_data = data.copy()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        self.metadata = metadata

        self.model = GaussianCopulaSynthesizer(metadata)
        self.model.fit(data)

        self.is_fitted = True

        synthetic_dir = resolve_synthetic_dir(data_dir, synthetic_dir)

        synthetic_data = self.sample(len(data))
        x_output, y_output = save_synthetic_split(data, synthetic_data, synthetic_dir)

        print("Gaussian Copula synthetic data saved:")
        print(f"X -> {x_output}")
        print(f"y -> {y_output}")

        return self

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError("Model must be trained before sampling")

        if n is None:
            n = len(self.train_data)

        if n <= 0:
            raise ValueError("Number of rows must be greater than zero")

        synthetic_data = self.model.sample(num_rows=n)

        return synthetic_data

    def evaluate(self, *args, **kwargs) -> float:

        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")

        return 0.0