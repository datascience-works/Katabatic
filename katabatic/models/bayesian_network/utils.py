"""Discretization helpers for the Bayesian Network model.

pgmpy's discrete Bayesian networks need every variable to take a finite set of
states, so numeric columns are quantile-binned into `n_bins` buckets before
structure/parameter learning, and decoded back to a representative value
(the bin midpoint) when sampling synthetic rows.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_numeric_cols(dataset_dir: str, df: pd.DataFrame) -> List[str]:
    """Return the subset of `df`'s columns that are genuinely numeric.

    Prefers the dataset's info.json (authoritative — integer-coded categorical
    columns, e.g. `car`'s, can't be told apart from numeric ones by dtype
    alone). Falls back to dtype-based detection if info.json is missing.
    """
    info_path = os.path.join(dataset_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cols = list(df.columns)
        return [cols[i] for i in info.get("num_col_idx", []) if i < len(cols)]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


class Discretizer:
    def __init__(self, num_cols: List[str], n_bins: int = 10):
        self.num_cols = num_cols
        self.n_bins = n_bins
        self._edges: Dict[str, np.ndarray] = {}
        self._midpoints: Dict[str, np.ndarray] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.num_cols:
            binned, edges = pd.qcut(
                df[c], q=self.n_bins, duplicates="drop", retbins=True
            )
            codes = binned.cat.codes.replace(-1, 0)
            self._edges[c] = edges
            self._midpoints[c] = (edges[:-1] + edges[1:]) / 2
            out[c] = codes.astype(str)
        for c in df.columns:
            if c not in self.num_cols:
                out[c] = df[c].astype(str)
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.num_cols:
            mids = self._midpoints[c]
            idx = out[c].astype(int).clip(0, len(mids) - 1)
            out[c] = mids[idx.to_numpy()]
        return out
