import numpy as np
import pandas as pd


def is_integer_valued(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = series.dropna().to_numpy(dtype=float)
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        return False
    return bool(np.all(vals == np.round(vals)))


def infer_feature_types(
    df: pd.DataFrame,
    features: list,
    categorical_cols=None,
    continuous_cols=None,
    max_int_categories: int = 20,
) -> dict:
    types = {}
    cat_set = set(categorical_cols) if categorical_cols else set()
    cont_set = set(continuous_cols) if continuous_cols else set()

    for col in features:
        if col in cat_set:
            types[col] = "categorical"
        elif col in cont_set:
            types[col] = "continuous"
        else:
            s = df[col]
            if (
                s.dtype == "object"
                or isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(s)
            ):
                types[col] = "categorical"
            elif is_integer_valued(s) and s.nunique(dropna=True) <= max_int_categories:
                types[col] = "categorical"
            else:
                types[col] = "continuous"

    return types
