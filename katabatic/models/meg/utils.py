"""
Utility functions for MEG model preprocessing and encoding.

Provides schema inference, tabular encoding/decoding, and feature span generation
to support mixed-type tabular data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class Schema:
    """
    Data schema describing tabular structure after preprocessing.

    Stores column types, categorical mappings, and encoded feature layout.
    """
    columns: List[str]
    cat_cols: List[str]
    num_cols: List[str]
    cat_values: Dict[str, List[str]]
    cat_blocks: Dict[str, Tuple[int, int]]
    d_enc: int


def infer_schema(X: pd.DataFrame) -> Schema:
    """
    Infer schema from a raw dataframe.

    Identifies categorical and numerical columns, and builds encoding layout.
    """
    cols = list(X.columns)
    cat_cols, num_cols = [], []
    cat_values: Dict[str, List[str]] = {}

    for c in cols:
        if X[c].dtype == "object" or str(X[c].dtype).startswith("category"):
            cat_cols.append(c)
            cat_values[c] = sorted(X[c].astype(str).fillna("NA").unique().tolist())
        else:
            num_cols.append(c)

    cat_blocks: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    # Assign index ranges for each column in encoded space
    for c in cols:
        if c in cat_cols:
            k = len(cat_values[c])
            cat_blocks[c] = (cursor, cursor + k)
            cursor += k
        else:
            cursor += 1

    return Schema(
        columns=cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        cat_values=cat_values,
        cat_blocks=cat_blocks,
        d_enc=cursor,
    )


def encode_df(X: pd.DataFrame, schema: Schema) -> np.ndarray:
    """
    Encode dataframe into numerical format.

    Categorical features are converted to one-hot encoding,
    while numerical features are cast to float.
    """
    Xc = X.copy()

    # Standardize categorical and numerical values
    for c in schema.cat_cols:
        Xc[c] = Xc[c].astype(str).fillna("NA")
    for c in schema.num_cols:
        Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(0.0)

    parts = []

    # Build encoded representation column-wise
    for c in schema.columns:
        if c in schema.cat_cols:
            cats = schema.cat_values[c]
            mapping = {v: i for i, v in enumerate(cats)}

            onehot = np.zeros((len(Xc), len(cats)), dtype=np.float32)
            vals = Xc[c].astype(str).values

            for i, v in enumerate(vals):
                j = mapping.get(v, mapping.get("NA", None))
                if j is not None:
                    onehot[i, j] = 1.0

            parts.append(onehot)
        else:
            parts.append(Xc[c].values.astype(np.float32).reshape(-1, 1))

    return np.concatenate(parts, axis=1).astype(np.float32)


def decode_df(X_enc: np.ndarray, schema: Schema) -> pd.DataFrame:
    """
    Decode encoded array back into original dataframe format.

    Converts one-hot categorical blocks into labels and restores numerical columns.
    """
    data: Dict[str, np.ndarray] = {}
    cursor = 0

    for c in schema.columns:
        if c in schema.cat_cols:
            s, e = schema.cat_blocks[c]
            block = X_enc[:, s:e]

            idx = np.argmax(block, axis=1)
            cats = schema.cat_values[c]

            data[c] = np.array(
                [cats[i] if 0 <= i < len(cats) else cats[0] for i in idx],
                dtype=object
            )
            cursor = e
        else:
            data[c] = X_enc[:, cursor]
            cursor += 1

    return pd.DataFrame(data)


def make_spans(schema: Schema) -> List[Tuple[int, int]]:
    """
    Create feature spans used for masked training.

    Each categorical column is treated as a block, while numerical features
    are treated as individual spans.
    """
    spans: List[Tuple[int, int]] = []
    used = set()

    # Add categorical blocks
    for c in schema.columns:
        if c in schema.cat_cols:
            s, e = schema.cat_blocks[c]
            spans.append((s, e))
            for i in range(s, e):
                used.add(i)

    # Add numerical features as single-dimension spans
    for i in range(schema.d_enc):
        if i not in used:
            spans.append((i, i + 1))

    return spans