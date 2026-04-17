import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from katabatic.utils.column_types import is_numerical


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, na_values='?')
    df.dropna(axis=1, how='all', inplace=True)
    return df


def process_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    for col in df.columns:
        if is_numerical(df[col]):
            try:
                col_data = pd.to_numeric(df[col], errors='coerce')
                col_data = col_data.fillna(col_data.median())

                unique_vals = col_data.nunique(dropna=True)
                if unique_vals <= 1:
                    print(f"[!] Skipping column '{col}' — constant or all NaN")
                    continue

                df_copy[col] = col_data

            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                continue

    return df_copy


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df.columns:
        if not is_numerical(df_copy[col]):
            try:
                df_copy[col] = df_copy[col].astype(str).str.strip()
                df_copy[col] = df_copy[col].replace(['missing'], 'Missing')
                df_copy[col] = df_copy[col].fillna('Missing').astype(str)

                # Sort values, ensuring 'Missing' comes first
                unique_vals = sorted(
                    df_copy[col].unique(), key=lambda x: (x != 'Missing', x))

                le = LabelEncoder()
                le.classes_ = np.array(unique_vals)
                df_copy[col] = le.transform(df_copy[col])

            except Exception as e:
                print(f"Skipping categorical column '{col}' due to error: {e}")
    return df_copy


def encode_preprocess(file_path: str, output_path: str, target_col: str = None) -> None:
    """
    Load, clean and encode a raw CSV for use with the evaluation pipeline.

    Numerical columns are cleaned and NaN-filled with their median.
    Categorical columns are label-encoded with a stable sort (Missing first).
    All columns are renamed to 0..N so the pipeline can reference them by index.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file.
    output_path : str
        Path to write the processed CSV.
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

    # Keep a reference to the raw X (before encoding) to build reverse mappings
    X_raw = X.copy()

    X = process_numerical_columns(X)
    X = encode_categorical_columns(X)

    y = y.fillna('Missing').astype(str)
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    y_encoded = pd.Series(y_encoded, name=y.name)

    df_processed = pd.concat([X, y_encoded], axis=1)
    df_processed.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to: {output_path}")

    # Build and save reverse mappings so synthetic output can be decoded back
    # to human-readable column names and original category labels.
    mappings = {"categorical_encodings": {}}

    for col in X_raw.columns:
        if not is_numerical(X_raw[col]):
            col_data = X_raw[col].astype(str).str.strip()
            col_data = col_data.replace(['missing'], 'Missing')
            col_data = col_data.fillna('Missing')
            unique_vals = sorted(col_data.unique(), key=lambda x: (x != 'Missing', x))
            mappings["categorical_encodings"][col] = {
                str(j): v for j, v in enumerate(unique_vals)
            }

    # Target column — LabelEncoder sorts alphabetically
    mappings["categorical_encodings"][y.name] = {
        str(j): cls for j, cls in enumerate(y_encoder.classes_)
    }

    mappings_path = output_path.replace(".csv", "_mappings.json")
    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved column mappings to  : {mappings_path}")
