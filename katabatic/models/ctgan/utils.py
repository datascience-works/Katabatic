
import pandas as pd

def to_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)

def check_data(X):
    return not pd.isnull(X).values.any()
