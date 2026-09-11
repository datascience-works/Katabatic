import json
import os
import platform
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning

# Hide MLP convergence warnings.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)

# Hide rare-class cross-validation warnings.
warnings.filterwarnings(
    "ignore",
    message=r"The least populated class in y has only .*",
    category=UserWarning,
    module=r"sklearn\.model_selection\._split",
)
# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

for directory in (PROJECT_ROOT, BENCHMARKS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.registry import get_model  # noqa: E402

# ============================================================
# Shuttle configuration
# ============================================================

SEED = 42
REGRESSOR = "xgboost"

CATEGORICAL_COLUMNS: list[str] = []

NUMERICAL_COLUMNS = [
    "time",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
]

TARGET_COLUMN = "class"


# ============================================================
# Prepare model data
# ============================================================


def prepare_forest_data(
    train_df: pd.DataFrame,
    target_col: str,
    model_data_dir: Path,
) -> tuple[dict[str, list[str]], list[int], list[int]]:
    """Prepare Shuttle features and encode the target."""
    required_columns = [
        *NUMERICAL_COLUMNS,
        target_col,
    ]

    missing_columns = [
        column for column in required_columns if column not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {train_df.columns.tolist()}"
        )

    encoded_df = train_df.loc[:, required_columns].copy()
    category_maps: dict[str, list[str]] = {}

    # Treat the class column as categorical.
    target_values = train_df[target_col].astype("string").str.strip().fillna("Unknown")

    target_categories = sorted(target_values.unique().tolist())

    target_mapping = {
        category: index for index, category in enumerate(target_categories)
    }

    encoded_df[target_col] = target_values.map(target_mapping).astype(float)

    category_maps[target_col] = target_categories

    # Prepare numerical features.
    for column in NUMERICAL_COLUMNS:
        values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = values.median()

        if pd.isna(median):
            raise ValueError(f"Column '{column}' has no valid values.")

        encoded_df[column] = values.fillna(median).astype(float)

    encoded_df = encoded_df.astype(float)

    if not np.isfinite(encoded_df.to_numpy()).all():
        raise ValueError("Encoded data contains NaN or infinity.")

    model_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Target is included for joint generation.
    encoded_df.to_csv(
        model_data_dir / "x_train.csv",
        index=False,
    )

    with (model_data_dir / "category_maps.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            category_maps,
            file,
            indent=2,
        )

    categorical_indexes = [encoded_df.columns.get_loc(target_col)]

    integer_indexes = [
        encoded_df.columns.get_loc(column) for column in NUMERICAL_COLUMNS
    ]

    return (
        category_maps,
        categorical_indexes,
        integer_indexes,
    )


# ============================================================
# Decode generated data
# ============================================================


def decode_forest_output(
    generated_df: pd.DataFrame,
    train_df: pd.DataFrame,
    category_maps: dict[str, list[str]],
) -> pd.DataFrame:
    """Restore target labels and integer feature values."""
    synthetic_df = generated_df.copy()

    missing_columns = [
        column for column in train_df.columns if column not in synthetic_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Generated data is missing columns: {missing_columns}")

    # Restore target classes.
    for column, categories in category_maps.items():
        values = pd.to_numeric(
            synthetic_df[column],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Generated target '{column}' contains invalid values.")

        codes = np.clip(
            np.rint(values),
            0,
            len(categories) - 1,
        ).astype(int)

        synthetic_df[column] = np.asarray(categories)[codes]

    # Restore integer feature values.
    for column in NUMERICAL_COLUMNS:
        values = pd.to_numeric(
            synthetic_df[column],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Generated column '{column}' contains invalid values.")

        synthetic_df[column] = np.rint(values).astype(np.int64)

    return synthetic_df.loc[:, train_df.columns]


# ============================================================
# System summary
# ============================================================


def print_system_summary() -> None:
    """Print system and hardware information."""
    memory = psutil.virtual_memory()

    print("\nSystem summary")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Logical CPUs: {os.cpu_count()}")
    print(f"Total RAM: {memory.total / 1024**3:.2f} GiB")
    print(f"Available RAM: {memory.available / 1024**3:.2f} GiB")
    print(f"Regressor: {REGRESSOR}")
    print("Training device: CPU")


# ============================================================
# Main pipeline
# ============================================================


def main() -> None:
    start_time = perf_counter()

    config = RunConfig(
        dataset_name="shuttle",
        model_name="forest_diffusion",
        categorical_cols=CATEGORICAL_COLUMNS,
        continuous_cols=NUMERICAL_COLUMNS,
        target_col_raw=TARGET_COLUMN,
        constraints={},
    )

    print_system_summary()

    print("\nSTEP 1 — Preprocess and split Shuttle")

    train_df, test_df, target_col, paths = preprocess_and_split(config)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")
    print(f"Target column: {target_col}")

    model_data_dir = Path(paths["split_dir"]) / "forest_diffusion_joint"

    model_output_dir = Path(paths["synthetic_dir"]) / "forest_diffusion_joint"

    print("\nSTEP 2 — Encode features and target")

    (
        category_maps,
        categorical_indexes,
        integer_indexes,
    ) = prepare_forest_data(
        train_df=train_df,
        target_col=target_col,
        model_data_dir=model_data_dir,
    )

    print(f"Categorical indexes: {categorical_indexes}")
    print(f"Integer indexes: {integer_indexes}")

    if REGRESSOR == "xgboost":
        import xgboost

        print(f"XGBoost version: {xgboost.__version__}")

    print("\nSTEP 3 — Train Forest Diffusion")

    model = get_model(
        "forest_diffusion",
        regressor=REGRESSOR,
        n_t=20,
        n_steps=20,
        n_epochs=3,
        pairs_per_epoch=20_000,
        n_estimators=100,
        max_depth=7,
        eta=0.1,
        cat_indexes=categorical_indexes,
        int_indexes=integer_indexes,
        bin_indexes=[],
        max_train_rows=None,
        n_jobs=4,
        seed=SEED,
        reject_training_duplicates=True,
        max_resample_rounds=30,
        sampling_noise=0.01,
    )

    # Required so sample() returns original
    # target labels.
    model.category_maps = category_maps

    training_start = perf_counter()

    model.train(
        dataset_dir=str(model_data_dir),
        synthetic_dir=str(model_output_dir),
    )

    # Set again for stability evaluation.
    model.category_maps = category_maps

    training_seconds = perf_counter() - training_start

    print("\nSTEP 4 — Decode and save synthetic data")

    generated_path = model_output_dir / "x_synth.csv"

    if not generated_path.exists():
        raise FileNotFoundError(f"Generated file not found: {generated_path}")

    generated_df = pd.read_csv(generated_path)

    if len(generated_df) != len(train_df):
        raise ValueError(
            f"Expected {len(train_df)} rows, but generated {len(generated_df)}."
        )

    synthetic_df = decode_forest_output(
        generated_df=generated_df,
        train_df=train_df,
        category_maps=category_maps,
    )

    synthetic_df = save_synthetic(
        synthetic_df,
        train_df,
        paths,
        categorical_cols=config.categorical_cols,
    )

    print(f"Synthetic shape: {synthetic_df.shape}")

    print("\nSTEP 5 — Check model.sample()")

    test_sample = model.sample(
        1000,
        seed=SEED,
    )

    if not isinstance(test_sample, pd.DataFrame):
        raise TypeError("model.sample() must return a DataFrame.")

    real_classes = set(train_df[target_col].astype(str).str.strip().unique())

    synthetic_classes = set(test_sample[target_col].astype(str).str.strip().unique())

    matched_classes = real_classes & synthetic_classes

    print(f"Target classes matched: {len(matched_classes)}/{len(real_classes)}")
    print(f"Real classes: {sorted(real_classes)}")
    print(f"Synthetic classes: {sorted(synthetic_classes)}")

    print("\nSTEP 6 — Evaluate synthetic data")

    evaluate(
        model,
        config,
        train_df,
        synthetic_df,
        target_col,
        paths,
        test_df,
    )

    elapsed_seconds = perf_counter() - start_time

    print("\nRuntime summary")
    print(f"Training and generation: {training_seconds:.2f} seconds")
    print(f"Complete pipeline: {elapsed_seconds:.2f} seconds")
    print(f"Synthetic directory: {paths['synthetic_dir']}")
    print(f"Results directory: {paths['results_dir']}")


if __name__ == "__main__":
    main()
