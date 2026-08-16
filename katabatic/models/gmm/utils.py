"""Utilities shared by the Gmm model: dataset schema loading and encode/decode
between raw columns (numeric + categorical) and a purely numeric matrix that
sklearn's GaussianMixture can fit.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_column_roles(dataset_dir: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (categorical_cols, numeric_cols) for the feature dataframe `df`.

    Prefers the dataset's info.json (authoritative — integer-coded categorical
    columns like `car`'s can't be told apart from numeric ones by dtype alone).
    Falls back to dtype-based detection if info.json is missing.
    """
    info_path = os.path.join(dataset_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cols = list(df.columns)
        cat_cols = [cols[i] for i in info.get("cat_col_idx", []) if i < len(cols)]
        num_cols = [cols[i] for i in info.get("num_col_idx", []) if i < len(cols)]
        # Anything not explicitly listed (e.g. columns beyond target removal)
        # falls back to dtype detection so we never silently drop a column.
        listed = set(cat_cols) | set(num_cols)
        for c in cols:
            if c not in listed:
                if pd.api.types.is_numeric_dtype(df[c]):
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
        return cat_cols, num_cols

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cat_cols, num_cols


class TabularEncoder:
    """Encodes a mixed-type DataFrame into a single numeric matrix (label-encoding
    categorical columns) and back, preserving column order and dtypes.
    """

    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.columns = num_cols + cat_cols
        self._encoders: Dict[str, LabelEncoder] = {}
        self._cat_ranges: Dict[str, Tuple[int, int]] = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [df[self.num_cols].to_numpy(dtype=float)] if self.num_cols else []
        for c in self.cat_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(df[c].astype(str))
            self._encoders[c] = le
            self._cat_ranges[c] = (0, len(le.classes_) - 1)
            blocks.append(encoded.reshape(-1, 1).astype(float))
        if not blocks:
            raise ValueError("No columns to encode.")
        return np.concatenate(blocks, axis=1)

    def inverse_transform(self, matrix: np.ndarray) -> pd.DataFrame:
        n_num = len(self.num_cols)
        out = {}
        if n_num:
            num_block = matrix[:, :n_num]
            for i, c in enumerate(self.num_cols):
                out[c] = num_block[:, i]
        cat_block = matrix[:, n_num:]
        for i, c in enumerate(self.cat_cols):
            lo, hi = self._cat_ranges[c]
            idx = np.rint(cat_block[:, i]).clip(lo, hi).astype(int)
            out[c] = self._encoders[c].inverse_transform(idx)
        return pd.DataFrame(out, columns=self.columns)
