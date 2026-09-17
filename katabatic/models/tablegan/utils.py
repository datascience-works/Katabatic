from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ColumnMeta:
    """Metadata required to transform a table column."""

    name: str
    kind: str  # "continuous" or "categorical"
    minimum: float | None = None
    maximum: float | None = None
    categories: list[str] | None = None
    original_dtype: str | None = None


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Infer categorical columns using dtype and cardinality heuristics."""

    categorical: list[str] = []
    n_rows = max(len(df), 1)

    for column in df.columns:
        series = df[column]

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical.append(column)
            continue

        if pd.api.types.is_integer_dtype(series):
            unique_count = series.nunique(dropna=True)

            if unique_count <= 20 or unique_count / n_rows < 0.05:
                categorical.append(column)

    return categorical


def build_schema(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
) -> list[ColumnMeta]:
    """Build transformation metadata for a dataframe."""

    if categorical_cols is None and continuous_cols is None:
        categorical = set(infer_categorical_columns(df))
    else:
        categorical = set(categorical_cols or [])

    continuous = set(continuous_cols or [])

    unknown = (categorical | continuous) - set(df.columns)
    if unknown:
        raise ValueError(
            f"Configured columns are not present in the training data: "
            f"{sorted(unknown)}"
        )

    overlap = categorical & continuous
    if overlap:
        raise ValueError(
            f"Columns cannot be both categorical and continuous: {sorted(overlap)}"
        )

    schema: list[ColumnMeta] = []

    for column in df.columns:
        series = df[column]

        if column in categorical:
            categories = sorted(series.dropna().astype(str).unique().tolist())

            schema.append(
                ColumnMeta(
                    name=column,
                    kind="categorical",
                    categories=categories,
                    original_dtype=str(series.dtype),
                )
            )
            continue

        if column in continuous or (
            categorical_cols is None and continuous_cols is None
        ):
            numeric = pd.to_numeric(series, errors="raise")

            schema.append(
                ColumnMeta(
                    name=column,
                    kind="continuous",
                    minimum=float(numeric.min()),
                    maximum=float(numeric.max()),
                    original_dtype=str(series.dtype),
                )
            )
            continue

        # If benchmark metadata only explicitly identifies some columns,
        # classify remaining numeric columns as continuous.
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="raise")

            schema.append(
                ColumnMeta(
                    name=column,
                    kind="continuous",
                    minimum=float(numeric.min()),
                    maximum=float(numeric.max()),
                    original_dtype=str(series.dtype),
                )
            )
        else:
            categories = sorted(series.dropna().astype(str).unique().tolist())

            schema.append(
                ColumnMeta(
                    name=column,
                    kind="categorical",
                    categories=categories,
                    original_dtype=str(series.dtype),
                )
            )

    return schema


def encode_dataframe(
    df: pd.DataFrame,
    schema: list[ColumnMeta],
) -> np.ndarray:
    """Encode a dataframe into a numeric matrix.

    Continuous columns are scaled to [-1, 1].
    Categorical columns are represented as integer category positions
    scaled to [-1, 1].
    """

    encoded = np.zeros((len(df), len(schema)), dtype=np.float32)

    for index, meta in enumerate(schema):
        if meta.kind == "continuous":
            values = pd.to_numeric(df[meta.name], errors="raise").to_numpy(
                dtype=np.float32
            )

            minimum = float(meta.minimum)
            maximum = float(meta.maximum)

            if np.isclose(maximum, minimum):
                encoded[:, index] = 0.0
            else:
                encoded[:, index] = 2.0 * (values - minimum) / (maximum - minimum) - 1.0

        else:
            categories = meta.categories or []
            mapping = {
                category: category_index
                for category_index, category in enumerate(categories)
            }

            indices = df[meta.name].astype(str).map(mapping).to_numpy(dtype=np.float32)

            if len(categories) <= 1:
                encoded[:, index] = 0.0
            else:
                encoded[:, index] = 2.0 * indices / (len(categories) - 1) - 1.0

    return encoded


def decode_dataframe(
    encoded: np.ndarray,
    schema: list[ColumnMeta],
) -> pd.DataFrame:
    """Convert TableGAN's numeric representation back into a dataframe."""

    if encoded.ndim != 2:
        raise ValueError("Encoded data must be a two-dimensional array.")

    if encoded.shape[1] != len(schema):
        raise ValueError("Encoded column count does not match the fitted schema.")

    output: dict[str, np.ndarray | list[str]] = {}

    for index, meta in enumerate(schema):
        values = np.clip(encoded[:, index], -1.0, 1.0)

        if meta.kind == "continuous":
            minimum = float(meta.minimum)
            maximum = float(meta.maximum)

            if np.isclose(maximum, minimum):
                decoded = np.full(len(values), minimum)
            else:
                decoded = (values + 1.0) / 2.0 * (maximum - minimum) + minimum

            if meta.original_dtype and meta.original_dtype.startswith("int"):
                decoded = np.rint(decoded).astype(int)

            output[meta.name] = decoded

        else:
            categories = meta.categories or []

            if not categories:
                output[meta.name] = [None] * len(values)
                continue

            if len(categories) == 1:
                indices = np.zeros(len(values), dtype=int)
            else:
                indices = np.rint((values + 1.0) / 2.0 * (len(categories) - 1)).astype(
                    int
                )

                indices = np.clip(indices, 0, len(categories) - 1)

            output[meta.name] = [categories[i] for i in indices]

    return pd.DataFrame(output, columns=[meta.name for meta in schema])


def calculate_side_length(n_columns: int) -> int:
    """Return the smallest square side that can contain all columns."""

    if n_columns < 1:
        raise ValueError("TableGAN requires at least one column.")

    return int(np.ceil(np.sqrt(n_columns)))


def table_to_square(
    encoded: np.ndarray,
    side_length: int | None = None,
) -> np.ndarray:
    """Pad each encoded row and reshape it into a square matrix."""

    if encoded.ndim != 2:
        raise ValueError("Encoded data must be a two-dimensional array.")

    rows, columns = encoded.shape

    side = calculate_side_length(columns) if side_length is None else int(side_length)

    if side * side < columns:
        raise ValueError("side_length is too small for the number of table columns.")

    padded = np.zeros((rows, side * side), dtype=np.float32)
    padded[:, :columns] = encoded

    return padded.reshape(rows, 1, side, side)


def square_to_table(
    matrices: np.ndarray,
    n_columns: int,
) -> np.ndarray:
    """Flatten generated square matrices and remove padding."""

    if matrices.ndim != 4 or matrices.shape[1] != 1:
        raise ValueError(
            "Expected generated data with shape (rows, 1, side_length, side_length)."
        )

    flattened = matrices.reshape(matrices.shape[0], -1)

    if n_columns > flattened.shape[1]:
        raise ValueError("Requested table has more columns than the generated matrix.")

    return flattened[:, :n_columns].astype(np.float32)
