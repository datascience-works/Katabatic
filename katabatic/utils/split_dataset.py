import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from katabatic.utils.train_test_consistency import sanity_check_train_test


def compute_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified split on the last column as label. Sets y_train/y_test .name to label column name.
    """
    np.random.seed(seed)
    random.seed(seed)

    y = df.iloc[:, -1]
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    label_name = df.columns[-1]
    y_train.name = label_name
    y_test.name = label_name
    return df_train, df_test, X_train, y_train, X_test, y_test


def split_dataset(input_csv, output_dir, test_size=0.2, seed=42, *args, **kwargs):
    # Set seed globally
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded data with shape: {df.shape}")

    df_train, df_test, X_train, y_train, X_test, y_test = compute_train_test_split(
        df, test_size=test_size, seed=seed
    )

    # Save full datasets
    df_train.to_csv(os.path.join(output_dir, "train_full.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_full.csv"), index=False)
    print("Saved train/test full data")
    print(f"Train size: {df_train.shape}, Test size: {df_test.shape}")

    # Print class distribution to confirm stratification
    print("Train label distribution:\n", y_train.value_counts(normalize=True))
    print("Test label distribution:\n", y_test.value_counts(normalize=True))

    X_train.to_csv(os.path.join(output_dir, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    X_test.to_csv(os.path.join(output_dir, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
    print("Saved X/y split")
    print("Training shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)


def split_dataset_presplit(
    train_csv,
    test_csv,
    output_dir,
    *args,
    **kwargs,
):
    """Write train/test CSV layout under output_dir from existing split files (no random split)."""
    np.random.seed(kwargs.get("seed", 42))
    random.seed(kwargs.get("seed", 42))

    os.makedirs(output_dir, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    sanity_check_train_test(df_train, df_test)

    label_name = df_train.columns[-1]
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
    y_train.name = label_name
    y_test.name = label_name

    print(f"Loaded presplit train {df_train.shape}, test {df_test.shape}")

    df_train.to_csv(os.path.join(output_dir, "train_full.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_full.csv"), index=False)
    print("Saved train/test full data")

    print("Train label distribution:\n", y_train.value_counts(normalize=True))
    print("Test label distribution:\n", y_test.value_counts(normalize=True))

    X_train.to_csv(os.path.join(output_dir, "x_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    X_test.to_csv(os.path.join(output_dir, "x_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
    print("Saved X/y split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset for benchmarking tabular data generation models"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to preprocessed CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save split datasets"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion for test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )

    args = parser.parse_args()

    split_dataset(args.input_csv, args.output_dir, args.test_size, args.seed)
