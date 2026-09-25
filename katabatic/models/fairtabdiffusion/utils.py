"""Schema inference and encode/decode helpers for FairTabDiffusion.

Mirrors the approach used in ``katabatic/models/ctgan/utils.py``: continuous
columns are quantile-normalized to a standard Normal, categorical columns
(including string/object labels) are one-hot encoded. This keeps
FairTabDiffusion compatible with the shared ``benchmarks/runner.py``
preprocessing pipeline, which does not assume pre-discretized integer data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


@dataclass
class ColumnMeta:
    name: str
    kind: str  # 'continuous' | 'categorical'
    categories: list[str] | None = None
    qt: QuantileTransformer | None = None


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic categorical-column detection (dtype + cardinality)."""
    cat_cols: list[str] = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)
        if dt == "object" or dt.startswith("category"):
            cat_cols.append(col)
            continue
        if dt.startswith("int"):
            nunique = s.nunique(dropna=True)
            if nunique <= 20 or nunique / n_rows < 0.05:
                cat_cols.append(col)
    return cat_cols


def infer_schema(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
) -> list[ColumnMeta]:
    """Build a ColumnMeta schema for the given columns.

    If ``categorical_cols``/``continuous_cols`` are provided (as passed by
    ``benchmarks/runner.py``), they take precedence; otherwise columns are
    classified automatically via ``infer_categorical_columns``.
    """
    if categorical_cols is not None:
        cat_cols = set(categorical_cols)
    else:
        cat_cols = set(infer_categorical_columns(df))

    schema: list[ColumnMeta] = []
    for c in df.columns:
        if continuous_cols is not None and c in continuous_cols:
            is_cat = False
        elif categorical_cols is not None:
            is_cat = c in cat_cols
        else:
            is_cat = c in cat_cols

        if is_cat:
            cats = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            schema.append(ColumnMeta(name=c, kind="categorical", categories=cats))
        else:
            schema.append(ColumnMeta(name=c, kind="continuous"))
    return schema


def fit_transformers(df: pd.DataFrame, schema: list[ColumnMeta]) -> None:
    """Fit a QuantileTransformer (normal output) in-place for continuous cols."""
    for col in schema:
        if col.kind == "continuous":
            n = max(min(len(df), 1000), 10)
            qt = QuantileTransformer(
                output_distribution="normal", n_quantiles=n, random_state=0
            )
            qt.fit(df[[col.name]].astype(float))
            col.qt = qt


def encode_columns(
    df: pd.DataFrame, schema: list[ColumnMeta]
) -> tuple[np.ndarray, dict[str, tuple[int, int]], list[str]]:
    """Encode the given columns into a single continuous ndarray.

    Continuous columns become a single quantile-normalized column.
    Categorical columns become one-hot blocks. Returns the encoded array,
    a mapping of column name -> (start, end) slice in the array, and the
    processing order of column names.
    """
    blocks: dict[str, tuple[int, int]] = {}
    order: list[str] = []
    parts: list[np.ndarray] = []
    ptr = 0
    for col in schema:
        if col.kind == "continuous":
            vals = df[[col.name]].astype(float)
            enc = col.qt.transform(vals) if col.qt is not None else vals.to_numpy()
            parts.append(enc.astype(np.float32))
            blocks[col.name] = (ptr, ptr + 1)
            ptr += 1
        else:
            cats = col.categories or []
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            for row_idx, v in enumerate(df[col.name].astype(str).tolist()):
                idx = cat_to_idx.get(v)
                if idx is not None:
                    one_hot[row_idx, idx] = 1.0
            parts.append(one_hot)
            blocks[col.name] = (ptr, ptr + len(cats))
            ptr += len(cats)
        order.append(col.name)
    return np.concatenate(parts, axis=1), blocks, order


def decode_columns(
    arr: np.ndarray, schema: list[ColumnMeta], blocks: dict[str, tuple[int, int]]
) -> pd.DataFrame:
    """Inverse of ``encode_columns``: continuous via inverse quantile transform,
    categorical via argmax over the one-hot block."""
    out: dict[str, list] = {}
    for col in schema:
        start, end = blocks[col.name]
        block = arr[:, start:end]
        if col.kind == "continuous":
            if col.qt is not None:
                out[col.name] = col.qt.inverse_transform(block).ravel()
            else:
                out[col.name] = block.ravel()
        else:
            cats = col.categories or []
            idx = block.argmax(axis=1)
            out[col.name] = [cats[i] if i < len(cats) else None for i in idx]
    return pd.DataFrame(out)
