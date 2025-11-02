from __future__ import annotations

from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder


def infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Infer categorical columns using dtype heuristics and cardinality.

    Rules:
    - object or category dtypes are categorical
    - integer columns with very low cardinality (<= 20 and < 5% of rows)
      are treated as categorical
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
            if nunique <= 20 and nunique / n_rows < 0.05:
                cat_cols.append(col)

    return cat_cols


@dataclass
class ColumnMeta:
    name: str
    kind: str  # 'continuous' | 'categorical'
    categories: List[str] | None = None
    qt: QuantileTransformer | None = None


def infer_schema(df: pd.DataFrame) -> List[ColumnMeta]:
    schema: List[ColumnMeta] = []
    cat_cols = set(infer_categorical_columns(df))
    for c in df.columns:
        if c in cat_cols:
            cats = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            schema.append(ColumnMeta(
                name=c, kind='categorical', categories=cats))
        else:
            schema.append(ColumnMeta(name=c, kind='continuous'))
    return schema


def fit_transformers(df: pd.DataFrame, schema: List[ColumnMeta]) -> None:
    for col in schema:
        if col.kind == 'continuous':
            qt = QuantileTransformer(
                output_distribution='normal', n_quantiles=min(1000, max(len(df), 10)))
            vals = df[[col.name]].astype(float)
            qt.fit(vals)
            col.qt = qt
        else:
            # categories already stored (sorted strings)
            pass


def encode_df(df: pd.DataFrame, schema: List[ColumnMeta]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
    """Encode dataframe to a continuous + one-hot representation.

    Returns: (encoded_array, cat_blocks, output_order)
    - cat_blocks: mapping col_name -> (start, end) indices of its one-hot block
    - output_order: list of original column names in order encountered
    """
    arrays: List[np.ndarray] = []
    cat_blocks: Dict[str, Tuple[int, int]] = {}
    output_order: List[str] = []

    for col in schema:
        output_order.append(col.name)
        if col.kind == 'continuous':
            vals = df[[col.name]].astype(float).values
            if col.qt is not None:
                vals = col.qt.transform(vals)
            arrays.append(vals.astype(np.float32))
        else:
            cats = col.categories or []
            # Build mapping str(value)->index
            cat_to_idx = {v: i for i, v in enumerate(cats)}
            idxs = df[col.name].astype(str).map(
                lambda v: cat_to_idx.get(v, -1)).values
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = idxs >= 0
            one_hot[np.where(valid)[0], idxs[valid]] = 1.0
            start = int(sum(a.shape[1] for a in arrays))
            arrays.append(one_hot)
            end = int(sum(a.shape[1] for a in arrays))
            cat_blocks[col.name] = (start, end)

    enc = np.concatenate(arrays, axis=1) if arrays else np.zeros(
        (len(df), 0), dtype=np.float32)
    return enc, cat_blocks, output_order


def decode_batch(enc: np.ndarray, schema: List[ColumnMeta]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    col_ptr = 0
    for i in range(enc.shape[0]):
        row: Dict[str, object] = {}
        col_ptr = 0
        for col in schema:
            if col.kind == 'continuous':
                val = float(enc[i, col_ptr])
                if col.qt is not None:
                    val = float(col.qt.inverse_transform(
                        np.array([[val]]))[0, 0])
                row[col.name] = val
                col_ptr += 1
            else:
                size = len(col.categories or [])
                block = enc[i, col_ptr:col_ptr+size]
                idx = int(np.argmax(block)) if size > 0 else -1
                cats = col.categories or []
                row[col.name] = cats[idx] if 0 <= idx < len(cats) else None
                col_ptr += size
        rows.append(row)
    return pd.DataFrame(rows)


def build_conditioning(df: pd.DataFrame, schema: List[ColumnMeta]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Build a full one-hot condition space across all categorical columns.

    Returns: (cond_matrix for each row based on actual categories, cond_blocks mapping)
    """
    arrays: List[np.ndarray] = []
    cond_blocks: Dict[str, Tuple[int, int]] = {}
    for col in schema:
        if col.kind == 'categorical':
            cats = col.categories or []
            cat_to_idx = {v: i for i, v in enumerate(cats)}
            idxs = df[col.name].astype(str).map(
                lambda v: cat_to_idx.get(v, -1)).values
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = idxs >= 0
            one_hot[np.where(valid)[0], idxs[valid]] = 1.0
            start = int(sum(a.shape[1] for a in arrays))
            arrays.append(one_hot)
            end = int(sum(a.shape[1] for a in arrays))
            cond_blocks[col.name] = (start, end)
    cond = np.concatenate(arrays, axis=1) if arrays else np.zeros(
        (len(df), 0), dtype=np.float32)
    return cond, cond_blocks


def sample_conditions(n: int, schema: List[ColumnMeta], cond_blocks: Dict[str, Tuple[int, int]],
                      empirical_probs: Dict[str, np.ndarray] | None = None,
                      device: str | None = None) -> np.ndarray:
    """Sample condition vectors by choosing one categorical column uniformly and a category per its distribution.
    """
    total_dim = 0
    spans: List[Tuple[str, int]] = []
    for col in schema:
        if col.kind == 'categorical':
            dim = len(col.categories or [])
            spans.append((col.name, dim))
            total_dim += dim
    if total_dim == 0:
        return np.zeros((n, 0), dtype=np.float32)

    conds = np.zeros((n, total_dim), dtype=np.float32)
    for i in range(n):
        # choose a categorical column uniformly
        j = np.random.randint(len(spans))
        name, dim = spans[j]
        start, end = cond_blocks[name]
        if empirical_probs and name in empirical_probs:
            p = empirical_probs[name]
            k = int(np.random.choice(len(p), p=p))
        else:
            k = int(np.random.randint(dim))
        conds[i, start + k] = 1.0
    return conds.astype(np.float32)
