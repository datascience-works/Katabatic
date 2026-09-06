from __future__ import annotations

import os
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def load_split_dataset(
    data_dir: str,
) -> tuple[pd.DataFrame, pd.Series]:

    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Expected x_train.csv and y_train.csv in: {data_dir}"
        )

    x_train = pd.read_csv(x_path)
    y_frame = pd.read_csv(y_path)

    if y_frame.shape[1] != 1:
        raise ValueError(
            "y_train.csv must contain exactly one target column."
        )

    if len(x_train) != len(y_frame):
        raise ValueError(
            "x_train.csv and y_train.csv must have the same row count."
        )

    y_train = y_frame.iloc[:, 0].copy()
    y_train.name = y_frame.columns[0]

    return x_train, y_train


def prepare_arf_dataframe(
    x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Combine X and y into the full table learned by ARF.

    arfpy requires categorical variables to use pandas category dtype.
    The target is included in the joint distribution so ARF learns
    relationships between features and the target.
    """
    x_frame = pd.DataFrame(x).copy().reset_index(drop=True)
    y_series = pd.Series(y).copy().reset_index(drop=True)

    target_col = y_series.name or "target"
    y_series.name = target_col

    if target_col in x_frame.columns:
        raise ValueError(
            f"Target column '{target_col}' already exists in X."
        )

    if len(x_frame) != len(y_series):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    data = pd.concat([x_frame, y_series], axis=1)

    categorical = list(categorical_cols or [])

    unknown = [
        col for col in categorical
        if col not in x_frame.columns
    ]

    if unknown:
        raise ValueError(
            f"Categorical columns not found in X: {unknown}"
        )

    # Explicit categorical features
    for col in categorical:
        data[col] = data[col].astype("category")

    # Classification target is categorical
    data[target_col] = data[target_col].astype("category")

    # Automatically protect string/object columns
    for col in x_frame.columns:
        if (
            pd.api.types.is_object_dtype(data[col])
            or pd.api.types.is_bool_dtype(data[col])
        ):
            data[col] = data[col].astype("category")

            if col not in categorical:
                categorical.append(col)

    if data.isna().any().any():
        missing = data.columns[
            data.isna().any()
        ].tolist()

        raise ValueError(
            "ARF input contains missing values in columns: "
            f"{missing}"
        )

    return data, target_col, categorical


def split_synthetic(
    synthetic_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split full synthetic data into synthetic X and y."""

    if target_col not in synthetic_df.columns:
        raise ValueError(
            f"Synthetic data is missing target column '{target_col}'."
        )

    x_synth = synthetic_df.drop(
        columns=[target_col]
    ).copy()

    y_synth = synthetic_df[
        target_col
    ].copy()

    y_synth.name = target_col

    return x_synth, y_synth