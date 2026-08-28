from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer categorical columns using dtype + low-cardinality integer heuristic.

    Rules:
    - object or category => categorical
    - int columns with low cardinality (<= 20 and < 5% rows) => categorical
    """
    cat_cols: List[str] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)

        if dt == "object" or dt.startswith("category"):
            cat_cols.append(col)
            continue

        if dt.startswith("int"):
            nunique = s.nunique(dropna=True)
            if nunique <= 20 and (nunique / n_rows) < 0.05:
                cat_cols.append(col)

    return cat_cols



# Schema

@dataclass
class ColumnMeta:
    name: str
    kind: str  # "continuous" | "categorical"
    categories: Optional[List[str]] = None  # for categorical
    mean: Optional[float] = None            # for continuous (z-score)
    std: Optional[float] = None             # for continuous (z-score)


def infer_schema(df: pd.DataFrame) -> List[ColumnMeta]:
    schema: List[ColumnMeta] = []
    cat_cols = set(infer_categorical_columns(df))

    for c in df.columns:
        if c in cat_cols:
            cats = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            schema.append(ColumnMeta(name=c, kind="categorical", categories=cats))
        else:
            schema.append(ColumnMeta(name=c, kind="continuous"))

    return schema


def fit_schema_stats(df: pd.DataFrame, schema: List[ColumnMeta]) -> None:
    """
    Fit mean/std for continuous columns for z-score standardization.
    """
    for col in schema:
        if col.kind == "continuous":
            vals = pd.to_numeric(df[col.name], errors="coerce").astype(float)
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            if s < 1e-8:
                s = 1.0  # avoid divide-by-zero
            col.mean = m
            col.std = s


# Encode / Decode

def encode_df(
    df: pd.DataFrame,
    schema: List[ColumnMeta],
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]], List[str]]:
    """
    Encode df -> numeric matrix suitable for TabEBM.

    - continuous: z-score standardization
    - categorical: ordinal encoding (0..K-1)
      (We keep it ordinal because TabEBM does gradient sampling; one-hot is unstable)

    Returns:
      enc: (n, d) float32
      cat_maps: mapping col -> {"cat_str": idx}
      col_order: original column order
    """
    arrays: List[np.ndarray] = []
    cat_maps: Dict[str, Dict[str, int]] = {}
    col_order: List[str] = []

    for col in schema:
        col_order.append(col.name)

        if col.kind == "continuous":
            vals = pd.to_numeric(df[col.name], errors="coerce").astype(float).values.reshape(-1, 1)
            m = col.mean if col.mean is not None else float(np.nanmean(vals))
            s = col.std if col.std is not None else float(np.nanstd(vals))
            if s < 1e-8:
                s = 1.0
            z = (vals - m) / s
            arrays.append(z.astype(np.float32))

        else:
            cats = col.categories or []
            mapping = {v: i for i, v in enumerate(cats)}
            cat_maps[col.name] = mapping

            idxs = df[col.name].astype(str).map(lambda v: mapping.get(v, -1)).values.astype(np.int64)

            # unknown -> random valid category (keeps values in-range)
            if len(cats) > 0:
                unk = np.where(idxs < 0)[0]
                if len(unk) > 0:
                    idxs[unk] = np.random.randint(0, len(cats), size=len(unk))

            arrays.append(idxs.reshape(-1, 1).astype(np.float32))

    enc = np.concatenate(arrays, axis=1) if arrays else np.zeros((len(df), 0), dtype=np.float32)
    return enc, cat_maps, col_order


def decode_df(
    enc: np.ndarray,
    schema: List[ColumnMeta],
) -> pd.DataFrame:
    """
    Decode numeric matrix back to dataframe in original feature space.

    - continuous: inverse z-score
    - categorical: round -> clip -> map idx -> category string
    """
    rows: List[Dict[str, object]] = []

    for i in range(enc.shape[0]):
        row: Dict[str, object] = {}
        ptr = 0

        for col in schema:
            if col.kind == "continuous":
                z = float(enc[i, ptr])
                m = col.mean if col.mean is not None else 0.0
                s = col.std if col.std is not None else 1.0
                val = z * s + m
                row[col.name] = float(val)
                ptr += 1
            else:
                cats = col.categories or []
                if len(cats) == 0:
                    row[col.name] = None
                else:
                    raw = float(enc[i, ptr])
                    idx = int(np.round(raw))
                    idx = int(np.clip(idx, 0, len(cats) - 1))
                    row[col.name] = cats[idx]
                ptr += 1

        rows.append(row)

    return pd.DataFrame(rows)


#  Convenience: full pipeline

def fit_and_encode(df: pd.DataFrame) -> Tuple[np.ndarray, List[ColumnMeta], List[str]]:
    """
    Convenience helper:
    - infer schema
    - fit mean/std
    - encode
    Returns (X_encoded, schema, col_order)
    """
    schema = infer_schema(df)
    fit_schema_stats(df, schema)
    X_enc, _, col_order = encode_df(df, schema)
    return X_enc, schema, col_order
