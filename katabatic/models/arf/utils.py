from __future__ import annotations

import os
from typing import Tuple, Optional
import pandas as pd


def load_train_df(data_dir: str) -> pd.DataFrame:
    """
    Load training data in the typical Katabatic style.

    Supported:
    - train_full.csv (preferred)
    - x_train.csv + y_train.csv (fallback)

    Returns a single dataframe containing features + label as the last column.
    """
    train_full = os.path.join(data_dir, "train_full.csv")
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        return pd.read_csv(train_full)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Could not find training files in {data_dir}. "
            f"Expected train_full.csv OR x_train.csv + y_train.csv."
        )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    if y.shape[1] != 1:
        raise ValueError("y_train.csv must have exactly one column (the target label).")

    y_col = y.columns[0]
    return pd.concat([X, y[[y_col]]], axis=1)


def split_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Split a full dataframe into X and y.
    Assumption: label is last column (common in Katabatic).
    """
    label = df.columns[-1]
    X = df.iloc[:, :-1].copy()
    y = df[label].copy()
    return X, y, label


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def try_align_columns(data_dir: str, X_synth: pd.DataFrame) -> pd.DataFrame:
    """
    Align synthetic feature columns to match x_train.csv columns (prevents mismatch issues).
    If x_train.csv exists, we force same column order and names.
    """
    x_path = os.path.join(data_dir, "x_train.csv")
    if not os.path.exists(x_path):
        return X_synth

    try:
        real_cols = pd.read_csv(x_path, nrows=0).columns.tolist()
        # Only align if shapes match
        if len(real_cols) == X_synth.shape[1]:
            X_synth = X_synth.copy()
            X_synth.columns = real_cols
            X_synth = X_synth.reindex(columns=real_cols)
    except Exception:
        pass

    return X_synth
