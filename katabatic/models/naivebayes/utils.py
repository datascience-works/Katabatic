import pandas as pd


def infer_feature_types(df: pd.DataFrame, features: list, categorical_cols=None, continuous_cols=None) -> dict:
    """
    Determine categorical vs continuous for each feature column.

    Prefers explicit categorical_cols/continuous_cols if provided (e.g. from
    RunConfig, as used by benchmarks/examples/*.py scripts), since dtype
    alone can't distinguish pre-encoded integer categoricals (e.g. the Car
    dataset, where all features are integer-coded) from genuinely continuous
    columns. Falls back to dtype-based detection for any column not covered.
    """
    types = {}
    cat_set = set(categorical_cols) if categorical_cols else set()
    cont_set = set(continuous_cols) if continuous_cols else set()

    for col in features:
        if col in cat_set:
            types[col] = "categorical"
        elif col in cont_set:
            types[col] = "continuous"
        else:
            dtype = df[col].dtype
            if dtype == "object" or str(dtype).startswith("category"):
                types[col] = "categorical"
            else:
                types[col] = "continuous"

    return types