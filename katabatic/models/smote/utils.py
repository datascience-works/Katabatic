"""
Utility functions for the SMOTE model pipeline.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

def load_training_data(data_dir: str) -> pd.DataFrame:
    """
    Looks for train_full.csv first, then x_train.csv + y_train.csv.

    Args:
        data_dir: Path to training data directory.

    Returns:
        DataFrame with the labels in the last column.

    Raises:
        FileNotFoundError: If no training data files are found.
        ValueError: The file y_train.csv must be only one column.
    """
    train_full = os.path.join(data_dir, "train_full.csv")
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        return pd.read_csv(train_full)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Could not find training data in {data_dir}.")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    if y.shape[1] != 1:
        raise ValueError("y_train.csv must have exactly one column.")

    y_col = y.columns[0]
    return pd.concat([X, y[y_col]], axis=1)


def save_synthetic_data(
    X_final: np.ndarray,
    y_final: np.ndarray,
    column_names: list[str],
    label: str,
    synth_dir: str,
) -> tuple[str, str]:
    """
    Save synthetic features and labels to CSV files.

    Args:
        X_final: Final synthetic feature array.
        y_final: Final synthetic label array.
        column_names: All column names (features + label).
        label: Name of the label column.
        synth_dir: Directory to save output files.

    Returns:
        Tuple of (x_path_out, y_path_out) file paths.
    """
    os.makedirs(synth_dir, exist_ok=True)

    df_synth = pd.DataFrame(
        np.column_stack([X_final, y_final]), columns=column_names
    )

    feature_cols = [c for c in column_names if c != label]
    x_synth = df_synth[feature_cols].copy()
    y_synth = df_synth[[label]].copy()

    x_path_out = os.path.join(synth_dir, "x_synth.csv")
    y_path_out = os.path.join(synth_dir, "y_synth.csv")

    x_synth.to_csv(x_path_out, index=False)
    y_synth.to_csv(y_path_out, index=False, header=True)

    return x_path_out, y_path_out


def save_metadata(
    synth_dir: str,
    df: pd.DataFrame,
    label: str,
    adjusted_k: int,
    sampling_strategy: str,
    n_original: int,
    n_synthetic: int,
    n_returned: int,
) -> None:
    """
    Save training metadata to a JSON file in the synthetic output directory.

    Args:
        synth_dir: Directory to save metadata.json.
        df: Original training DataFrame (used for schema info).
        label: Name of the label column.
        adjusted_k: k_neighbors value actually used.
        sampling_strategy: Sampling strategy used.
        n_original: Number of original training samples.
        n_synthetic: Number of synthetic samples generated.
        n_returned: Number of samples in the final output.
    """
    meta = {
        "schema": {
            "columns": df.columns.tolist(),
            "label": label,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        },
        "training": {
            "k_neighbors": adjusted_k,
            "sampling_strategy": sampling_strategy,
            "n_original": n_original,
            "n_synthetic": n_synthetic,
            "n_returned": n_returned,
        },
    }

    meta_path = os.path.join(synth_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def resolve_synth_dir(synthetic_dir: str | None, data_dir: str, model_name: str) -> str:
    """
    Resolve the output directory for synthetic data.
    Uses synthetic_dir if provided, otherwise builds a default path.

    Args:
        synthetic_dir: Explicitly provided output directory (can be None).
        data_dir: Input data directory (used to infer dataset name).
        model_name: Model name used in the default path ('smote').

    Returns:
        Resolved output directory path.
    """
    if synthetic_dir:
        return synthetic_dir

    dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
    return os.path.join("synthetic", dataset_name, model_name)