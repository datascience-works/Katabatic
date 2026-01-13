import pandas as pd

def infer_target_col(df: pd.DataFrame, preferred=("class", "target", "label", "y")) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    # fallback: last column (common in many CSVs)
    return df.columns[-1]

def split_xy(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
