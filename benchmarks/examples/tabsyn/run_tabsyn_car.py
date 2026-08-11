import json
import logging
import os
import platform
import sys
import warnings
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd
import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)
# ============================================================
# PROJECT PATH SETUP
# ============================================================

# Current script location:
# Katabatic/benchmarks/examples/tabsyn/run_tabsyn_car.py
#
# Moving up three directories reaches:
# Katabatic/
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, "benchmarks")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

print(f"Project root: {PROJECT_ROOT}")

# ============================================================
# PROJECT IMPORTS
# ============================================================

from runner import (
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)
from katabatic.models.tabsyn.models import TabSyn

logging.getLogger("pgmpy").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
# ============================================================
# RUNTIME REPORT
# ============================================================

def print_runtime_report(
    duration_seconds,
    start_timestamp,
    end_timestamp,
    model_name,
    dataset_name,
):
    """Print the total execution duration."""

    print("\n" + "=" * 70)
    print("Evaluation Runtime Report")
    print("=" * 70)
    print(f"Start time : {start_timestamp}")
    print(f"End time   : {end_timestamp}")
    print(
        f"{model_name} took {duration_seconds:.2f} seconds "
        f"to run on the {dataset_name} dataset."
    )
    print("=" * 70)


def print_system_report():
    """Print basic operating-system and PyTorch device information."""

    system_details = platform.uname()

    print("\n" + "=" * 70)
    print("Computation Hardware Summary")
    print("=" * 70)
    print(f"System    : {system_details.system}")
    print(f"Release   : {system_details.release}")
    print(f"Machine   : {system_details.machine}")
    print(f"Processor : {system_details.processor}")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device_name = "Apple Silicon GPU available, training uses CPU"

        else:
            device_name = "CPU"

        print(f"Device    : {device_name}")

    except Exception as error:
        print(f"Device    : unavailable ({error})")

    try:
        import psutil

        memory = psutil.virtual_memory()

        print(f"Total RAM : {memory.total / 1e9:.4f} GB")
        print(f"Free RAM  : {memory.available / 1e9:.4f} GB")
        print(f"Used RAM  : {memory.used / 1e9:.4f} GB")

    except ImportError:
        print("RAM       : psutil is not installed")

    print("=" * 70)


# ============================================================
# TABSyn INPUT PREPARATION
# ============================================================

def create_tabsyn_input_files(
    train_df,
    test_df,
    split_dir,
    categorical_columns,
    target_column,
):
    """
    Create the NumPy files expected by the TabSyn utility code.

    The Car dataset has no continuous columns, so X_num arrays are
    saved with shape (number_of_rows, 0).

    Files created:
        X_num_train.npy
        X_cat_train.npy
        y_train.npy
        X_num_test.npy
        X_cat_test.npy
        y_test.npy
        info.json
    """

    os.makedirs(split_dir, exist_ok=True)

    print("\nPreparing TabSyn input files...")

    # The Car dataset contains no continuous features.
    train_numeric = np.empty(
        (len(train_df), 0),
        dtype=np.float32,
    )

    test_numeric = np.empty(
        (len(test_df), 0),
        dtype=np.float32,
    )

    # Convert categorical features to strings.
    train_categorical = (
        train_df[categorical_columns]
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=str)
    )

    test_categorical = (
        test_df[categorical_columns]
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=str)
    )

    # Convert targets to strings.
    train_target = (
        train_df[target_column]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    test_target = (
        test_df[target_column]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    np.save(
        os.path.join(split_dir, "X_num_train.npy"),
        train_numeric,
    )

    np.save(
        os.path.join(split_dir, "X_cat_train.npy"),
        train_categorical,
    )

    np.save(
        os.path.join(split_dir, "y_train.npy"),
        train_target,
    )

    np.save(
        os.path.join(split_dir, "X_num_test.npy"),
        test_numeric,
    )

    np.save(
        os.path.join(split_dir, "X_cat_test.npy"),
        test_categorical,
    )

    np.save(
        os.path.join(split_dir, "y_test.npy"),
        test_target,
    )

    info = {
        "task_type": "multiclass",
        "n_classes": int(
            train_df[target_column].nunique()
        ),
        "dataset_name": "car",
        "continuous_columns": [],
        "categorical_columns": categorical_columns,
        "target_column": target_column,
    }

    info_path = os.path.join(
        split_dir,
        "info.json",
    )

    with open(
        info_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            info,
            file,
            indent=4,
        )

    print("TabSyn input files created successfully.")
    print(f"X_num_train shape : {train_numeric.shape}")
    print(f"X_cat_train shape : {train_categorical.shape}")
    print(f"y_train shape     : {train_target.shape}")
    print(f"X_num_test shape  : {test_numeric.shape}")
    print(f"X_cat_test shape  : {test_categorical.shape}")
    print(f"y_test shape      : {test_target.shape}")
    print(f"Metadata          : {info_path}")

# ============================================================
# CATEGORICAL DECODING
# ============================================================

def decode_categorical_columns(
    synthetic_features,
    train_df,
    categorical_columns,
):
    """
    Convert generated categorical indices back to original Car labels.

    TabSyn encodes each categorical column using sorted unique values.
    This function reconstructs the same ordering and maps generated
    integer indices back to their original categories.
    """

    decoded_df = synthetic_features.copy()

    for column in categorical_columns:
        if column not in decoded_df.columns:
            raise ValueError(
                f"Synthetic feature column '{column}' is missing.\n"
                f"Available columns: {decoded_df.columns.tolist()}"
            )

        categories = np.unique(
            train_df[column]
            .fillna("Unknown")
            .astype(str)
            .to_numpy()
        )

        if len(categories) == 0:
            raise ValueError(
                f"No categories were found for column '{column}'."
            )

        generated_indices = pd.to_numeric(
            decoded_df[column],
            errors="coerce",
        )

        # Replace invalid generated values with category index zero.
        generated_indices = (
            generated_indices
            .fillna(0)
            .round()
            .astype(int)
            .clip(
                lower=0,
                upper=len(categories) - 1,
            )
        )

        decoded_df[column] = [
            categories[index]
            for index in generated_indices
        ]

    return decoded_df

# ============================================================
# LOAD MODEL-GENERATED SYNTHETIC FILES
# ============================================================

def load_generated_synthetic_data(
    paths,
    train_df,
    categorical_columns,
    target_column,
):
    """
    Load x_synth.csv and y_synth.csv generated inside TabSyn.train().

    The feature indices are decoded to the original Car categories, and
    the feature and target files are combined into one DataFrame.
    """

    x_synthetic_path = os.path.join(
        paths["synthetic_dir"],
        "x_synth.csv",
    )

    y_synthetic_path = os.path.join(
        paths["synthetic_dir"],
        "y_synth.csv",
    )

    if not os.path.exists(x_synthetic_path):
        raise FileNotFoundError(
            "Synthetic feature file was not found:\n"
            f"{x_synthetic_path}"
        )

    if not os.path.exists(y_synthetic_path):
        raise FileNotFoundError(
            "Synthetic target file was not found:\n"
            f"{y_synthetic_path}"
        )

    synthetic_features = pd.read_csv(
        x_synthetic_path
    )

    synthetic_target = pd.read_csv(
        y_synthetic_path
    )

    print("\nGenerated feature file:")
    print(f"Path    : {x_synthetic_path}")
    print(f"Shape   : {synthetic_features.shape}")
    print(f"Columns : {synthetic_features.columns.tolist()}")
    print("\nGenerated target file:")
    print(f"Path    : {y_synthetic_path}")
    print(f"Shape   : {synthetic_target.shape}")
    print(f"Columns : {synthetic_target.columns.tolist()}")

    # The model normally renames feature columns using x_train.csv.
    # Handle cat_1...cat_6 as a fallback if renaming did not occur.
    expected_feature_columns = categorical_columns

    if not all(
        column in synthetic_features.columns
        for column in expected_feature_columns
    ):
        tabsyn_feature_columns = [
            f"cat_{index}"
            for index in range(
                1,
                len(categorical_columns) + 1,
            )
        ]

        if all(
            column in synthetic_features.columns
            for column in tabsyn_feature_columns
        ):
            synthetic_features = (
                synthetic_features[
                    tabsyn_feature_columns
                ].copy()
            )

            synthetic_features.columns = (
                categorical_columns
            )

        elif synthetic_features.shape[1] == len(
            categorical_columns
        ):
            synthetic_features.columns = (
                categorical_columns
            )

        else:
            raise ValueError(
                "Unable to align synthetic feature columns.\n"
                f"Expected: {categorical_columns}\n"
                f"Received: {synthetic_features.columns.tolist()}"
            )

    synthetic_features = decode_categorical_columns(
        synthetic_features=synthetic_features,
        train_df=train_df,
        categorical_columns=categorical_columns,
    )

    if synthetic_target.shape[1] != 1:
        raise ValueError(
            "Expected y_synth.csv to contain exactly one column, "
            f"but found {synthetic_target.shape[1]} columns."
        )

    synthetic_target.columns = [target_column]

    synthetic_target[target_column] = (
        synthetic_target[target_column]
        .fillna("Unknown")
        .astype(str)
    )

    synthetic_df = pd.concat(
        [
            synthetic_features.reset_index(drop=True),
            synthetic_target.reset_index(drop=True),
        ],
        axis=1,
    )

    missing_columns = [
        column
        for column in train_df.columns
        if column not in synthetic_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The synthetic dataset is missing columns:\n"
            f"{missing_columns}\n"
            f"Received columns: {synthetic_df.columns.tolist()}"
        )

    # Match the exact training-data column order.
    synthetic_df = synthetic_df[
        train_df.columns.tolist()
    ]

    return synthetic_df

# ============================================================
# MAIN
# ============================================================

def main():
    start_counter = perf_counter()
    start_timestamp = datetime.now()

    config = RunConfig(
        dataset_name="car",
        model_name="tabsyn",
        categorical_cols=["0","1","2","3","4","5",],
        continuous_cols=[],
        target_col_raw="6",
        constraints={},
        max_train_rows=None,
    )

    print("\n" + "=" * 60)
    print("CAR DATASET — TABSyn")
    print("=" * 60)
    print(f"Dataset             : {config.dataset_name}")
    print(f"Model               : {config.model_name}")
    print(f"Categorical columns : {config.categorical_cols}")
    print(f"Continuous columns  : {config.continuous_cols}")
    print(f"Raw target column   : {config.target_col_raw}")

    # ========================================================
    # STEP 1 AND STEP 2 — PREPROCESS AND SPLIT
    # ========================================================

    train_df, test_df, target_col, paths = (
        preprocess_and_split(config)
    )

    print("\nDataset preparation completed.")
    print(f"Training shape : {train_df.shape}")
    print(f"Testing shape  : {test_df.shape}")
    print(f"Target column  : {target_col}")
    print(f"Columns        : {train_df.columns.tolist()}")

    required_columns = (
        config.categorical_cols
        + [target_col]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Configured columns are missing from the processed dataset:\n"
            f"{missing_columns}\n"
            f"Available columns: {train_df.columns.tolist()}"
        )

    # Create only the NumPy files required by TabSyn utilities.
    create_tabsyn_input_files(
        train_df=train_df,
        test_df=test_df,
        split_dir=paths["split_dir"],
        categorical_columns=config.categorical_cols,
        target_column=target_col,
    )

    # ========================================================
    # STEP 3 — TRAIN TABSyn
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 3 — Train TabSyn")
    print("=" * 60)

    model = TabSyn(
        d_token=16,

        decoder_epochs=50,
        decoder_batch_size=256,

        diffusion_epochs=300,
        diffusion_batch_size=256,

        diffusion_steps=50,

        lr=1e-3,
        weight_decay=0.0,
        patience=20,

        seed=42,
        device="cpu",
    )

    model_save_dir = os.path.join(
        paths["results_dir"],
        "model_files",
    )

    # TabSyn.train() already:
    # 1. trains the decoder,
    # 2. trains the diffusion model,
    # 3. generates synthetic samples,
    # 4. saves x_synth.csv and y_synth.csv.
    model.train(
        data_dir=paths["split_dir"],
        save_dir=model_save_dir,
        synthetic_dir=paths["synthetic_dir"],
        extra_info={
            "categorical_cols": config.categorical_cols,
            "continuous_cols": config.continuous_cols,
            "target_col": target_col,
        },
    )

    print("\nTabSyn training and generation completed.")

    # ========================================================
    # STEP 4 — LOAD SYNTHETIC DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 4 — Load generated synthetic data")
    print("=" * 60)

    synthetic_df = load_generated_synthetic_data(
        paths=paths,
        train_df=train_df,
        categorical_columns=config.categorical_cols,
        target_column=target_col,
    )

    synthetic_df = save_synthetic(
        synthetic_df=synthetic_df,
        train_df=train_df,
        paths=paths,
        categorical_cols=config.categorical_cols,
    )

    # ========================================================
    # STEP 5 — EVALUATE
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 5 — Evaluate synthetic data")
    print("=" * 60)

    evaluate(
        model=model,
        config=config,
        train_df=train_df,
        synthetic_df=synthetic_df,
        target_col=target_col,
        paths=paths,
        test_df=test_df,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    end_timestamp = datetime.now()
    duration_seconds = (
        perf_counter() - start_counter
    )

    print_runtime_report(
        duration_seconds=duration_seconds,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
    )
    print_system_report()

if __name__ == "__main__":
    main()