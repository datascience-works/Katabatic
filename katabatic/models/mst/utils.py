"""Utility functions for the MST model pipeline."""

from __future__ import annotations

import json
import os

import pandas as pd


def load_training_data(data_dir: str) -> pd.DataFrame:
    """
    Load Katabatic training data.

    Looks for train_full.csv first, then x_train.csv + y_train.csv.
    """
    train_full = os.path.join(data_dir, "train_full.csv")
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        return pd.read_csv(train_full)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Could not find training data in {data_dir}. "
            "Expected train_full.csv or x_train.csv/y_train.csv."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    if y.shape[1] != 1:
        raise ValueError("y_train.csv must have exactly one column.")

    y_col = y.columns[0]

    return pd.concat(
        [X.reset_index(drop=True), y[[y_col]].reset_index(drop=True)],
        axis=1,
    )


def resolve_synth_dir(
    synthetic_dir: str | None,
    data_dir: str,
    model_name: str = "mst",
) -> str:
    """Resolve the directory used to save synthetic data."""
    if synthetic_dir:
        return synthetic_dir

    dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"

    return os.path.join(
        "synthetic",
        dataset_name,
        model_name,
    )


def save_synthetic_data(
    synthetic_df: pd.DataFrame,
    label: str,
    synth_dir: str,
) -> tuple[str, str]:
    """Save synthetic features and target to Katabatic output files."""
    os.makedirs(synth_dir, exist_ok=True)

    if label not in synthetic_df.columns:
        raise ValueError(
            f"Target column '{label}' not found in synthetic data."
        )

    feature_cols = [
        column
        for column in synthetic_df.columns
        if column != label
    ]

    x_path = os.path.join(synth_dir, "x_synth.csv")
    y_path = os.path.join(synth_dir, "y_synth.csv")

    synthetic_df[feature_cols].to_csv(
        x_path,
        index=False,
    )

    synthetic_df[[label]].to_csv(
        y_path,
        index=False,
    )

    return x_path, y_path


def save_metadata(
    synth_dir: str,
    df: pd.DataFrame,
    label: str,
    epsilon: float,
    delta: float,
    categorical_columns: list[str],
    n_generated: int,
) -> None:
    """Save MST training and generation metadata."""
    os.makedirs(synth_dir, exist_ok=True)

    metadata = {
        "model": "mst",
        "schema": {
            "columns": df.columns.tolist(),
            "label": label,
            "dtypes": {
                column: str(df[column].dtype)
                for column in df.columns
            },
            "categorical_columns": categorical_columns,
        },
        "privacy": {
            "epsilon": epsilon,
            "delta": delta,
        },
        "training": {
            "n_original": len(df),
            "n_generated": n_generated,
        },
    }

    metadata_path = os.path.join(
        synth_dir,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )
