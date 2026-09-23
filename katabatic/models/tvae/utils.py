"""Helper utilities for the TVAE model wrapper.

Kept separate from models.py to follow Katabatic's three-file model
package convention (__init__.py, models.py, utils.py), even though the
TVAE wrapper itself delegates most work to ctgan.TVAE directly.
"""

from __future__ import annotations

import os

import pandas as pd


def load_training_data(data_dir: str) -> pd.DataFrame:
    """Load training data from a Katabatic split directory.

    Supports both the combined train_full.csv format and the separate
    x_train.csv / y_train.csv format, matching the convention used by
    benchmarks/runner.py and other Katabatic models (e.g. FairTabDiffusion).
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
    return pd.concat([X, y[y_col]], axis=1)


def resolve_discrete_columns(
    df: pd.DataFrame,
    label_col: str,
    categorical_cols: list[str] | None,
) -> list[str]:
    """Build the list of discrete (categorical) columns for ctgan.TVAE.fit().

    Ensures the label column is always treated as discrete, since it is
    the classification target for the standard Katabatic benchmark
    datasets, and filters out any requested columns not present in df.
    """
    cat_cols_full = list(categorical_cols or [])
    if label_col not in cat_cols_full:
        cat_cols_full = [*cat_cols_full, label_col]
    return [c for c in cat_cols_full if c in df.columns]
