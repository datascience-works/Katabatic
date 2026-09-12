"""Utility functions for the AIM model pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ColumnEncoding:
    """Encoding information required to decode an AIM column."""

    kind: str
    values: list[Any] | None = None
    bin_edges: list[float] | None = None
    representatives: list[float] | None = None
    original_dtype: str | None = None


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

    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if y_df.shape[1] != 1:
        raise ValueError("y_train.csv must have exactly one column.")

    target_col = y_df.columns[0]

    return pd.concat(
        [
            x_df.reset_index(drop=True),
            y_df[[target_col]].reset_index(drop=True),
        ],
        axis=1,
    )


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Infer categorical columns from pandas dtypes.

    Object, category, and boolean columns are treated as categorical.
    """

    categorical_columns: list[str] = []

    for column in df.columns:
        dtype = df[column].dtype

        if (
            dtype == "object"
            or str(dtype).startswith("category")
            or str(dtype) == "bool"
        ):
            categorical_columns.append(column)

    return categorical_columns


def encode_dataframe(
    df: pd.DataFrame,
    categorical_columns: list[str],
    continuous_columns: list[str],
    *,
    max_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, ColumnEncoding], dict[str, int]]:
    """
    Encode a pandas DataFrame into the integer domain required by Private-PGM.

    Categorical columns are ordinal encoded using their observed values.
    Continuous columns are discretised into quantile-based bins.

    Returns
    -------
    encoded_df:
        DataFrame containing only integer state values.
    encodings:
        Per-column metadata required to decode synthetic samples.
    domain_sizes:
        Number of discrete states for each column.
    """

    if max_bins < 2:
        raise ValueError("max_bins must be at least 2.")

    overlap = set(categorical_columns) & set(continuous_columns)

    if overlap:
        raise ValueError(
            f"Columns cannot be both categorical and continuous: {sorted(overlap)}"
        )

    expected_columns = set(categorical_columns) | set(continuous_columns)
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Columns were not found in training data: {sorted(missing_columns)}"
        )

    if expected_columns != set(df.columns):
        unspecified = set(df.columns) - expected_columns
        raise ValueError(
            "Every training column must be classified as categorical or continuous. "
            f"Unspecified columns: {sorted(unspecified)}"
        )

    encoded_df = pd.DataFrame(index=df.index)
    encodings: dict[str, ColumnEncoding] = {}
    domain_sizes: dict[str, int] = {}

    categorical_set = set(categorical_columns)

    for column in df.columns:
        series = df[column]

        if column in categorical_set:
            values = pd.Index(series.unique()).tolist()

            if not values:
                raise ValueError(f"Column '{column}' has no values.")

            value_to_code = {value: index for index, value in enumerate(values)}

            encoded_df[column] = series.map(value_to_code).astype(int)

            encodings[column] = ColumnEncoding(
                kind="categorical",
                values=values,
                original_dtype=str(series.dtype),
            )
            domain_sizes[column] = len(values)

            continue

        numeric = pd.to_numeric(series, errors="raise")

        unique_count = int(numeric.nunique())

        if unique_count == 0:
            raise ValueError(f"Column '{column}' has no values.")

        if unique_count == 1:
            representative = float(numeric.iloc[0])

            encoded_df[column] = 0

            encodings[column] = ColumnEncoding(
                kind="continuous",
                bin_edges=None,
                representatives=[representative],
                original_dtype=str(series.dtype),
            )
            domain_sizes[column] = 1

            continue

        n_bins = min(max_bins, unique_count)

        codes, bin_edges = pd.qcut(
            numeric,
            q=n_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )

        codes = pd.Series(codes, index=df.index).astype(int)

        representatives: list[float] = []

        for code in range(int(codes.max()) + 1):
            values_in_bin = numeric[codes == code]

            if values_in_bin.empty:
                left = bin_edges[code]
                right = bin_edges[code + 1]
                representative = float((left + right) / 2.0)
            else:
                representative = float(values_in_bin.median())

            representatives.append(representative)

        encoded_df[column] = codes

        encodings[column] = ColumnEncoding(
            kind="continuous",
            bin_edges=[float(value) for value in bin_edges],
            representatives=representatives,
            original_dtype=str(series.dtype),
        )
        domain_sizes[column] = len(representatives)

    return encoded_df.astype(int), encodings, domain_sizes


def decode_dataframe(
    encoded_df: pd.DataFrame,
    encodings: dict[str, ColumnEncoding],
    column_order: list[str],
) -> pd.DataFrame:
    """Decode AIM integer states back into the original column representation."""

    decoded_df = pd.DataFrame(index=encoded_df.index)

    for column in column_order:
        if column not in encoded_df.columns:
            raise ValueError(
                f"Synthetic data does not contain expected column '{column}'."
            )

        if column not in encodings:
            raise ValueError(f"No encoding metadata found for column '{column}'.")

        encoding = encodings[column]
        codes = encoded_df[column].astype(int)

        if encoding.kind == "categorical":
            if encoding.values is None:
                raise ValueError(f"Categorical encoding for '{column}' has no values.")

            values = encoding.values

            if ((codes < 0) | (codes >= len(values))).any():
                raise ValueError(
                    f"Synthetic codes for '{column}' are outside its domain."
                )

            decoded = codes.map(lambda code: values[int(code)])

        elif encoding.kind == "continuous":
            if encoding.representatives is None:
                raise ValueError(
                    f"Continuous encoding for '{column}' has no representatives."
                )

            representatives = encoding.representatives

            if ((codes < 0) | (codes >= len(representatives))).any():
                raise ValueError(
                    f"Synthetic codes for '{column}' are outside its domain."
                )

            decoded = codes.map(lambda code: representatives[int(code)])

        else:
            raise ValueError(f"Unknown encoding kind '{encoding.kind}' for '{column}'.")

        if encoding.original_dtype is not None:
            try:
                decoded = decoded.astype(encoding.original_dtype)
            except (TypeError, ValueError):
                pass

        decoded_df[column] = decoded

    return decoded_df[column_order]


def resolve_synth_dir(
    synthetic_dir: str | None,
    data_dir: str,
    model_name: str = "aim",
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
    """Save synthetic feature and target data in Katabatic format."""

    os.makedirs(synth_dir, exist_ok=True)

    if label not in synthetic_df.columns:
        raise ValueError(f"Target column '{label}' not found in synthetic data.")

    feature_columns = [column for column in synthetic_df.columns if column != label]

    x_path = os.path.join(synth_dir, "x_synth.csv")
    y_path = os.path.join(synth_dir, "y_synth.csv")

    synthetic_df[feature_columns].to_csv(
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
    *,
    epsilon: float,
    delta: float,
    degree: int,
    max_bins: int,
    categorical_columns: list[str],
    continuous_columns: list[str],
    domain_sizes: dict[str, int],
    n_generated: int,
) -> None:
    """Save AIM training and generation metadata."""

    os.makedirs(synth_dir, exist_ok=True)

    metadata = {
        "model": "aim",
        "schema": {
            "columns": df.columns.tolist(),
            "label": label,
            "dtypes": {column: str(df[column].dtype) for column in df.columns},
            "categorical_columns": categorical_columns,
            "continuous_columns": continuous_columns,
            "domain_sizes": domain_sizes,
        },
        "privacy": {
            "epsilon": epsilon,
            "delta": delta,
        },
        "configuration": {
            "degree": degree,
            "max_bins": max_bins,
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
