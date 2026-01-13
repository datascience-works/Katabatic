from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def normalize(counts: np.ndarray) -> np.ndarray:
    """
    Normalize a non-negative array into a probability distribution.
    Falls back to uniform if sum is zero.
    """
    counts = np.asarray(counts, dtype=float)
    counts[counts < 0] = 0.0
    s = counts.sum()
    if s > 0:
        return counts / s
    return np.ones_like(counts) / max(1, counts.size)


def is_numeric(series: pd.Series) -> bool:
    """Check whether a pandas Series is numeric."""
    return pd.api.types.is_numeric_dtype(series)


def discretize_numeric_dataframe(
    df: pd.DataFrame, n_bins: int
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Discretise numeric columns into quantile bins [0..n_bins-1].

    Returns:
        discretised_df
        metadata: {col: (min, max)}
    """
    out = df.copy()
    meta: Dict[str, Tuple[float, float]] = {}

    for col in out.columns:
        if is_numeric(out[col]):
            values = out[col].astype(float)
            meta[col] = (float(values.min()), float(values.max()))
            try:
                bins = pd.qcut(values, q=n_bins, duplicates="drop")
                out[col] = bins.cat.codes.astype(int)
            except Exception:
                out[col] = 0

    return out, meta
