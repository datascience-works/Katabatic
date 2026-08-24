from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from katabatic.models.base_model import Model


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

        train_full = os.path.join(data_dir, "train_full.csv")
        x_train = os.path.join(data_dir, "x_train.csv")
        y_train = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            data = pd.read_csv(train_full)

        elif os.path.exists(x_train) and os.path.exists(y_train):
            x = pd.read_csv(x_train)
            y = pd.read_csv(y_train)

            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have one target column")

            data = pd.concat([x, y], axis=1)

        else:
            raise FileNotFoundError(
                "Training files were not found in the given folder"
            )

        self.train_data = data.copy()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        self.metadata = metadata

        self.model = GaussianCopulaSynthesizer(metadata)
        self.model.fit(data)

        self.is_fitted = True

        # Decide where the synthetic data will be saved
        if synthetic_dir is None:
            dataset_name = os.path.basename(os.path.normpath(data_dir))
            synthetic_dir = os.path.join(
                "synthetic",
                dataset_name,
                "gaussian_copula",
            )

        os.makedirs(synthetic_dir, exist_ok=True)

        # Generate the same number of synthetic rows as the training data
        synthetic_data = self.sample(len(data))

        # Assume the last column is the target column
        target_column = data.columns[-1]

        x_synth = synthetic_data[data.columns[:-1]].copy()
        y_synth = synthetic_data[[target_column]].copy()

        x_output = os.path.join(synthetic_dir, "x_synth.csv")
        y_output = os.path.join(synthetic_dir, "y_synth.csv")

        x_synth.to_csv(x_output, index=False)
        y_synth.to_csv(y_output, index=False)

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