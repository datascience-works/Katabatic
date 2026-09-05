import pandas as pd

from katabatic.utils.column_types import is_numerical

__all__ = [
    "fill_categorical_nulls",
    "load_and_clean_data",
    "preprocess_dataset",
    "preprocess_tabular",
    "process_numerical_columns",
]


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, na_values="?")
    df.dropna(axis=1, how="all", inplace=True)
    return df


def process_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    for col in df.columns:
        if not is_numerical(df[col]):
            continue
        try:
            col_data = pd.to_numeric(df[col], errors="coerce")
            col_data = col_data.fillna(col_data.median())

            unique_vals = col_data.nunique(dropna=True)
            if unique_vals <= 1:
                print(f"[!] Dropping column '{col}' — constant or all NaN")
                df_copy.drop(columns=[col], inplace=True)
                continue

            df_copy[col] = col_data

        except Exception as e:
            print(f"Error processing column '{col}': {e}")
            continue

    return df_copy


def fill_categorical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df.columns:
        if not is_numerical(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str).str.strip()
            df_copy[col] = df_copy[col].replace(
                {"nan": "Missing", "missing": "Missing"}
            )
    return df_copy


def preprocess_dataset(
    file_path: str, output_path: str, target_col: str = None
) -> None:
    """
    Load and clean a raw CSV. No encoding is applied — models handle their own
    data representation. Only the minimum necessary cleaning is done:

      - '?' values are treated as NaN on load
      - All-NaN columns are dropped
      - Numeric columns: NaN filled with column median; constant columns dropped
      - String columns: NaN / 'nan' normalised to the string 'Missing'

    The target column is moved to the last position but otherwise untouched.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file.
    output_path : str
        Path to write the cleaned CSV.
    target_col : str, optional
        Name of the target column. Falls back to the last column if omitted.
    """
    print(f"Preprocessing: {file_path}")
    df = load_and_clean_data(file_path)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X = process_numerical_columns(X)
    X = fill_categorical_nulls(X)

    # Keep target as-is; just normalise any NaN to the string 'Missing'
    y = y.fillna("Missing").astype(str).str.strip()

    df_processed = pd.concat([X, y], axis=1)
    df_processed.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to: {output_path}")


preprocess_tabular = preprocess_dataset
