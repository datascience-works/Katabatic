import pandas as pd
import numpy as np
import argparse
import os
import random
from sklearn.model_selection import train_test_split


def split_dataset(input_csv, output_dir, test_size=0.2, seed=42, *args, **kwargs):
    # Set seed globally
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded data with shape: {df.shape}")

    # Split just once
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=df.iloc[:, -1]
    )

    # Create all splits from the same indices
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]
    # df_train, df_test = df, df

    # Save full datasets
    df_train.to_csv(os.path.join(output_dir, 'train_full.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test_full.csv'), index=False)
    print("Saved train/test full data")
    print(f"Train size: {df_train.shape}, Test size: {df_test.shape}")

    # Save X/y splits
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    # Ensure label has a name
    y_train.name = df.columns[-1]
    y_test.name = df.columns[-1]

    # Print class distribution to confirm stratification
    print("Train label distribution:\n", y_train.value_counts(normalize=True))
    print("Test label distribution:\n", y_test.value_counts(normalize=True))

    X_train.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'),
                   index=False, header=True)
    X_test.to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'),
                  index=False, header=True)
    print("Saved X/y split")
    print("Training shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset for benchmarking tabular data generation models")
    parser.add_argument('--input_csv', type=str,
                        required=True, help="Path to preprocessed CSV")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save split datasets")
    parser.add_argument('--test_size', type=float,
                        default=0.2, help="Proportion for test set")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for splitting")

    args = parser.parse_args()

    split_dataset(
        args.input_csv,
        args.output_dir,
        args.test_size,
        args.seed
    )
