import pandas as pd


def is_numerical(col) -> bool:
    """Return True if the column has a numeric dtype (excluding booleans)."""
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def get_column_types(df: pd.DataFrame, exclude_last: bool = True):
    """
    Identify categorical and continuous columns in a DataFrame.

    This is the single source of truth for column type detection in Katabatic.
    Always call this on the **raw** DataFrame before any preprocessing, so that
    original dtypes are intact. Integer-encoded categorical columns cannot be
    distinguished from continuous columns after preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect. Pass the raw DataFrame before preprocess_dataset
        so that original string/numeric dtypes are preserved.
    exclude_last : bool
        If True (default), the last column is treated as the target and excluded
        from both lists. Set to False to include all columns.

    Returns
    -------
    categorical_cols : list[str]
        Original column names whose dtype is non-numeric.
    continuous_cols : list[str]
        Original column names whose dtype is numeric.

    Warnings
    --------
    Auto-detection based on dtype is a heuristic. Integer columns that represent
    categories (e.g. zip codes, encoded education levels) will be misclassified
    as continuous. Always pass column types explicitly when you know your data.
    """
    cols = list(df.columns[:-1]) if exclude_last else list(df.columns)
    categorical_cols = [c for c in cols if not is_numerical(df[c])]
    continuous_cols = [c for c in cols if is_numerical(df[c])]
    return categorical_cols, continuous_cols
