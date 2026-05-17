"""Utility functions for a self-contained Katabatic TVAE-style model.

No SDV dependency is used here. The helpers below handle:
- loading Katabatic train split files
- combining X/y into one training table
- mixed tabular encoding for numeric + categorical columns
- inverse transforming generated samples
- saving x_synth.csv and y_synth.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_TARGET_NAME = "target"


@dataclass
class ColumnSpec:
    name: str
    kind: str  # "numeric" or "categorical"
    start: int
    end: int
    mean: float | None = None
    std: float | None = None
    categories: list[str] | None = None
    was_integer: bool = False


class TabularPreprocessor:
    """Encode and decode mixed tabular data without SDV.

    Numeric columns are z-score scaled. Categorical columns are one-hot encoded.
    During inverse transform, categorical groups are reconstructed with argmax.
    """

    def __init__(self, categorical_columns: Optional[Iterable[str]] = None):
        self.categorical_columns = set(categorical_columns or [])
        self.columns: list[str] = []
        self.specs: list[ColumnSpec] = []
        self.output_dim: int = 0

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.columns = list(df.columns)
        self.specs = []
        cursor = 0

        for col in self.columns:
            series = df[col]
            is_categorical = col in self.categorical_columns

            if is_categorical:
                categories = pd.Series(series.astype(str).fillna("__nan__")).unique().tolist()
                if not categories:
                    categories = ["__nan__"]
                start, end = cursor, cursor + len(categories)
                self.specs.append(
                    ColumnSpec(
                        name=col,
                        kind="categorical",
                        start=start,
                        end=end,
                        categories=categories,
                    )
                )
                cursor = end
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                mean = float(numeric.mean()) if not np.isnan(numeric.mean()) else 0.0
                std = float(numeric.std(ddof=0)) if not np.isnan(numeric.std(ddof=0)) else 1.0
                if std == 0.0:
                    std = 1.0
                start, end = cursor, cursor + 1
                self.specs.append(
                    ColumnSpec(
                        name=col,
                        kind="numeric",
                        start=start,
                        end=end,
                        mean=mean,
                        std=std,
                        was_integer=pd.api.types.is_integer_dtype(series),
                    )
                )
                cursor = end

        self.output_dim = cursor
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.specs:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        arrays: list[np.ndarray] = []
        for spec in self.specs:
            series = df[spec.name]
            if spec.kind == "numeric":
                numeric = pd.to_numeric(series, errors="coerce").fillna(spec.mean)
                values = ((numeric.to_numpy(dtype=np.float32) - spec.mean) / spec.std).reshape(-1, 1)
                arrays.append(values)
            else:
                cats = spec.categories or []
                lookup = {value: idx for idx, value in enumerate(cats)}
                values = series.astype(str).fillna("__nan__").map(lookup).fillna(0).astype(int).to_numpy()
                one_hot = np.zeros((len(series), len(cats)), dtype=np.float32)
                one_hot[np.arange(len(series)), values] = 1.0
                arrays.append(one_hot)

        return np.concatenate(arrays, axis=1).astype(np.float32)

    def inverse_transform(self, matrix: np.ndarray) -> pd.DataFrame:
        if not self.specs:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform().")

        data: dict[str, np.ndarray | list[str]] = {}
        for spec in self.specs:
            block = matrix[:, spec.start : spec.end]
            if spec.kind == "numeric":
                values = block[:, 0] * spec.std + spec.mean
                if spec.was_integer:
                    values = np.rint(values).astype(int)
                data[spec.name] = values
            else:
                cats = spec.categories or ["__nan__"]
                idx = np.argmax(block, axis=1)
                decoded = [cats[int(i)] for i in idx]
                data[spec.name] = decoded

        return pd.DataFrame(data, columns=self.columns)


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_train_split(data_dir: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load x_train.csv and y_train.csv from a Katabatic split directory."""
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    x_train = _read_csv_if_exists(x_path)
    y_train_df = _read_csv_if_exists(y_path)

    if x_train is None or y_train_df is None:
        raise FileNotFoundError(
            "Could not find x_train.csv and y_train.csv in "
            f"{data_dir}. Make sure the TrainTestSplitPipeline created the split first."
        )
    if y_train_df.shape[1] == 0:
        raise ValueError("y_train.csv has no columns.")

    target_col = y_train_df.columns[0] if y_train_df.columns[0] else DEFAULT_TARGET_NAME
    return x_train, y_train_df.iloc[:, 0], target_col


def combine_features_target(x_train: pd.DataFrame, y_train: pd.Series, target_col: str) -> tuple[pd.DataFrame, str]:
    """Combine X and y while avoiding target-name clashes."""
    df = x_train.copy().reset_index(drop=True)
    clean_target_col = target_col
    if clean_target_col in df.columns:
        clean_target_col = DEFAULT_TARGET_NAME
    df[clean_target_col] = y_train.reset_index(drop=True)
    return df, clean_target_col


def infer_categorical_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Infer categorical columns for Katabatic classification datasets.

    Object/category/bool columns are categorical. The target is categorical.
    Small low-cardinality non-float columns are also treated as categorical,
    which helps datasets such as car/nursery when their categories are read as ints.
    """
    categorical_columns: list[str] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        unique_ratio = nunique / n_rows

        if (
            col == target_col
            or series.dtype == "object"
            or str(series.dtype) == "category"
            or series.dtype == "bool"
            or (not pd.api.types.is_float_dtype(series) and nunique <= 20 and unique_ratio <= 0.2)
        ):
            categorical_columns.append(col)

    return categorical_columns


def split_synthetic(synthetic_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if target_col not in synthetic_df.columns:
        target_col = synthetic_df.columns[-1]
    y_synth = synthetic_df[[target_col]].copy()
    x_synth = synthetic_df.drop(columns=[target_col]).copy()
    return x_synth, y_synth


def save_synthetic_outputs(x_synth: pd.DataFrame, y_synth: pd.DataFrame, synthetic_dir: str) -> None:
    os.makedirs(synthetic_dir, exist_ok=True)
    x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
    y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
