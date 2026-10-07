from __future__ import annotations

import os

import pandas as pd


def load_training_dataframe(data_dir: str) -> pd.DataFrame:
    """Load a training dataframe from either a combined CSV or split X/y files."""
    train_full = os.path.join(data_dir, "train_full.csv")
    x_train = os.path.join(data_dir, "x_train.csv")
    y_train = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        return pd.read_csv(train_full)

    if os.path.exists(x_train) and os.path.exists(y_train):
        x = pd.read_csv(x_train)
        y = pd.read_csv(y_train)

        if y.shape[1] != 1:
            raise ValueError("y_train.csv must have one target column")

        return pd.concat([x, y], axis=1)

    raise FileNotFoundError("Training files were not found in the given folder")


def resolve_synthetic_dir(data_dir: str, synthetic_dir: str | None = None) -> str:
    """Return the directory where synthetic data should be saved."""
    if synthetic_dir is not None:
        return synthetic_dir

    dataset_name = os.path.basename(os.path.normpath(data_dir))
    return os.path.join("synthetic", dataset_name, "gaussian_copula")


def save_synthetic_split(
    data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    synthetic_dir: str,
) -> tuple[str, str]:
    """Write split feature/target CSV files for synthetic data."""
    os.makedirs(synthetic_dir, exist_ok=True)

    target_column = data.columns[-1]
    x_synth = synthetic_data[data.columns[:-1]].copy()
    y_synth = synthetic_data[[target_column]].copy()

    x_output = os.path.join(synthetic_dir, "x_synth.csv")
    y_output = os.path.join(synthetic_dir, "y_synth.csv")

    x_synth.to_csv(x_output, index=False)
    y_synth.to_csv(y_output, index=False)

    return x_output, y_output
