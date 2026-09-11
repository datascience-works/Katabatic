"""
utils.py
--------
Data loading, validation, and preprocessing utilities for the Statlog
(Shuttle) dataset.

The dataset is expected as a CSV with the columns:
    time, a1, a2, a3, a4, a5, a6, a7, a8, class

- time, a1..a8 : integer-valued sensor/telemetry readings (features)
- class        : integer target label in {1, 2, 3, 4, 5, 6, 7}
                 (1 = "Rad Flow", the normal/majority state; the rest
                 are various fault/anomaly states)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: tuple[str, ...] = (
    "time",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
)
TARGET_COLUMN: str = "class"
ALL_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + (TARGET_COLUMN,)

CLASS_NAMES = {
    1: "Rad Flow",
    2: "Fpv Close",
    3: "Fpv Open",
    4: "High",
    5: "Bypass",
    6: "Bpv Close",
    7: "Bpv Open",
}


@dataclass
class Dataset:
    """Container bundling features/target plus a train/test split."""

    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.X.columns)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load the shuttle CSV file into a DataFrame and validate its shape.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")

    df = pd.read_csv(path)

    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    return df[list(ALL_COLUMNS)]


def validate_dataframe(df: pd.DataFrame) -> None:
    """Sanity-check dtypes, NaNs, and label range. Raises ValueError on failure."""
    if df[list(ALL_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    bad_labels = set(df[TARGET_COLUMN].unique()) - set(CLASS_NAMES.keys())
    if bad_labels:
        raise ValueError(f"Unexpected class labels found: {bad_labels}")


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return counts of each class label, sorted by label."""
    return df[TARGET_COLUMN].value_counts().sort_index()


def prepare_dataset(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    stratify: bool = True,
) -> Dataset:
    """
    Load the CSV, split into train/test, and (optionally) scale features.

    Parameters
    ----------
    path : str
        Path to shuttle.csv.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Seed for reproducibility.
    scale : bool
        If True, fit a StandardScaler on the training features and apply
        it to both train and test (scaler is returned on the Dataset).
    stratify : bool
        If True, stratify the split by class label (recommended given
        the strong class imbalance in this dataset).

    Returns
    -------
    Dataset
    """
    df = load_csv(path)
    validate_dataframe(df)

    X = df[list(FEATURE_COLUMNS)].copy()
    y = df[TARGET_COLUMN].copy()

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    scaler_obj: StandardScaler | None = None
    if scale:
        scaler_obj = StandardScaler()
        X_train = pd.DataFrame(
            scaler_obj.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler_obj.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    return Dataset(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler_obj,
    )


def scale_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a previously-fitted scaler to new feature data."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
