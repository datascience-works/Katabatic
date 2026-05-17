"""
Utility functions for the Katabatic SMOTE model.

This file keeps data loading, saving, metadata writing, and small preprocessing
helpers separate from the main model class.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_training_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Load Katabatic training data from either:
    - train_full.csv
    - x_train.csv and y_train.csv

    The label column is expected to be the final column.
    """
    train_full_path = os.path.join(data_dir, "train_full.csv")
    x_train_path = os.path.join(data_dir, "x_train.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full_path):
        return pd.read_csv(train_full_path)

    if not (os.path.exists(x_train_path) and os.path.exists(y_train_path)):
        raise FileNotFoundError(
            f"Could not find train_full.csv or x_train.csv/y_train.csv in {data_dir}."
        )

    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)

    if y.shape[1] != 1:
        raise ValueError("y_train.csv must contain exactly one label column.")

    return pd.concat([X, y], axis=1)


def split_features_label(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Split a dataframe into feature matrix, label array, and label column name.
    """
    if df.shape[1] < 2:
        raise ValueError("Training data must contain at least one feature and one label column.")

    label_col = df.columns[-1]
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y, label_col


def get_adjusted_k_neighbors(y: np.ndarray, requested_k: int) -> int:
    """
    SMOTE requires k_neighbors < number of samples in the smallest class.
    This function safely reduces k_neighbors when the minority class is small.
    """
    _, class_counts = np.unique(y, return_counts=True)
    min_class_size = int(class_counts.min())

    if min_class_size <= 1:
        raise ValueError(
            "SMOTE cannot be applied because the smallest class has only one sample. "
            "Use a different imbalance method or provide more minority-class examples."
        )

    if min_class_size <= requested_k:
        adjusted_k = max(1, min_class_size - 1)
        print(f"[SMOTE] Warning: smallest class has {min_class_size} samples.")
        print(f"[SMOTE] Adjusting k_neighbors from {requested_k} to {adjusted_k}.")
        return adjusted_k

    return requested_k


def resolve_synthetic_dir(data_dir: str, synthetic_dir: Optional[str]) -> str:
    """
    Resolve the output directory for synthetic data.
    """
    if synthetic_dir:
        return synthetic_dir

    dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
    return os.path.join("synthetic", dataset_name, "smote")


def build_synthetic_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """
    Combine feature and label arrays back into a dataframe using the original schema.
    """
    return pd.DataFrame(np.column_stack([X, y]), columns=columns)


def save_synthetic_outputs(
    df_synth: pd.DataFrame,
    label_col: str,
    output_dir: str,
) -> tuple[str, str]:
    """
    Save Katabatic-compatible x_synth.csv and y_synth.csv files.
    """
    os.makedirs(output_dir, exist_ok=True)

    x_synth = df_synth.drop(columns=[label_col]).copy()
    y_synth = df_synth[[label_col]].copy()

    x_path = os.path.join(output_dir, "x_synth.csv")
    y_path = os.path.join(output_dir, "y_synth.csv")

    x_synth.to_csv(x_path, index=False)
    y_synth.to_csv(y_path, index=False, header=True)

    return x_path, y_path


def save_metadata(
    output_dir: str,
    df_train: pd.DataFrame,
    label_col: str,
    k_neighbors: int,
    sampling_strategy: str,
    n_original: int,
    n_generated: int,
    n_returned: int,
) -> str:
    """
    Save metadata for reproducibility and debugging.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "model": "SMOTEModel",
        "schema": {
            "columns": df_train.columns.tolist(),
            "label": label_col,
            "dtypes": {column: str(df_train[column].dtype) for column in df_train.columns},
        },
        "training": {
            "k_neighbors": k_neighbors,
            "sampling_strategy": sampling_strategy,
            "n_original": int(n_original),
            "n_generated": int(n_generated),
            "n_returned": int(n_returned),
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
