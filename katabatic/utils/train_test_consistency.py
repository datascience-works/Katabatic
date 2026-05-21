from __future__ import annotations

import warnings

import pandas as pd


def sanity_check_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_mismatch_warn_tv: float = 0.25,
) -> None:
    """
    Validate presplit train/test tables: same columns and order, dtypes, non-empty,
    last column as label; warn if label marginals differ strongly (total variation).
    """
    if train_df.empty:
        raise ValueError("train DataFrame is empty")
    if test_df.empty:
        raise ValueError("test DataFrame is empty")

    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    if train_cols != test_cols:
        raise ValueError(
            f"Column mismatch: train has {train_cols!r}, test has {test_cols!r}"
        )

    for name in train_cols:
        if train_df[name].dtype != test_df[name].dtype:
            raise ValueError(
                f"dtype mismatch for column {name!r}: "
                f"train {train_df[name].dtype} vs test {test_df[name].dtype}"
            )

    y_tr = train_df.iloc[:, -1]
    y_te = test_df.iloc[:, -1]
    # Total variation distance between empirical label distributions
    all_labels = pd.unique(pd.concat([y_tr, y_te], ignore_index=True).dropna())
    p = y_tr.value_counts(normalize=True).reindex(all_labels, fill_value=0.0)
    q = y_te.value_counts(normalize=True).reindex(all_labels, fill_value=0.0)
    tv = float((p - q).abs().sum() / 2)
    if tv > label_mismatch_warn_tv:
        warnings.warn(
            f"Train/test label distributions differ (total variation ≈ {tv:.3f}); "
            "check stratification or leakage.",
            UserWarning,
            stacklevel=2,
        )
