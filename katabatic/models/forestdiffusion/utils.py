"""Shared preprocessing utilities for the ForestDiffusion Katabatic model.

CHANGE (repo-health / "remove duplicated code"):
The previous `encode_categorical_features` helper (OrdinalEncoder-based)
has been removed. It was dead code -- models.py never imported it and
instead re-implemented the same ordinal-encoding logic inline -- and it
also used the wrong encoding scheme for this method in the first place:
the source paper (Jolicoeur-Martineau et al., 2024, Section 3.6) encodes
categorical variables via one-hot/dummy encoding, not ordinal encoding.

The replacement below (`encode_categorical_onehot` / `decode_categorical_onehot`)
is now the single source of truth for categorical preprocessing and is
imported directly by models.py, instead of being duplicated inline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_categorical_onehot(
    x_train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[OneHotEncoder], List[str], List[str], int]:
    """One-hot encode all object-dtype columns of `x_train_df`.

    Numeric columns are kept as-is (cast to float32) and placed first;
    any object-dtype (categorical) columns are one-hot encoded and
    appended after them. This matches the paper's stated preprocessing:
    "we make the categorical variables continuous by dummy encoding them."

    Returns
    -------
    x_train : np.ndarray, shape (n, n_num_cols + n_onehot_cols)
        Numeric columns first, one-hot categorical columns after.
    encoder : OneHotEncoder or None
        Fitted encoder (None if there were no categorical columns).
    cat_cols : list[str]
        Names of the original categorical columns, in encoded order.
    num_cols : list[str]
        Names of the original numeric columns, in encoded order.
    n_num_cols : int
        Number of numeric columns (i.e. where the one-hot block starts).
    """
    # BUGFIX: columns were previously classified as categorical only when
    # dtype was exactly "object". That misses other non-numeric dtypes
    # (e.g. pandas' nullable StringDtype, or "category") and lets pure
    # text columns fall through into `num_cols`, which then crashes in
    # `.to_numpy(dtype=np.float32)` (e.g. car.csv's 'buying'/'safety'
    # columns, which contain values like 'med'). Detecting numeric
    # columns directly via `is_numeric_dtype` is robust to dtype
    # variations regardless of which string-like dtype pandas assigned.
    is_numeric = x_train_df.apply(
        lambda col: pd.api.types.is_numeric_dtype(col)
        and not pd.api.types.is_bool_dtype(col)
    )
    num_cols = list(x_train_df.columns[is_numeric])
    cat_cols = list(x_train_df.columns[~is_numeric])

    x_num = x_train_df[num_cols].to_numpy(dtype=np.float32)
    n_num_cols = x_num.shape[1]

    if len(cat_cols) == 0:
        return x_num, None, cat_cols, num_cols, n_num_cols

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_cat = encoder.fit_transform(x_train_df[cat_cols].astype(str)).astype(np.float32)

    x_train = np.concatenate([x_num, x_cat], axis=1)
    return x_train, encoder, cat_cols, num_cols, n_num_cols


def decode_categorical_onehot(
    x: np.ndarray,
    encoder: Optional[OneHotEncoder],
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    n_num_cols: int,
    feature_order: Sequence[str],
) -> pd.DataFrame:
    """Inverse of `encode_categorical_onehot`.

    Maps a generated (numeric + one-hot) matrix back to a DataFrame with
    the original column names, categorical labels, and column order.
    This is the inverse-encoding step that was previously entirely
    missing: generated categorical columns used to be written out as
    raw, un-decoded numeric codes.
    """
    df_num = pd.DataFrame(x[:, :n_num_cols], columns=list(num_cols))

    if encoder is None:
        return df_num[list(feature_order)]

    x_cat_onehot = x[:, n_num_cols:]
    x_cat = encoder.inverse_transform(x_cat_onehot)
    df_cat = pd.DataFrame(x_cat, columns=list(cat_cols))

    return pd.concat([df_num, df_cat], axis=1)[list(feature_order)]
