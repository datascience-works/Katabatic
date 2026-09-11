import json
import logging
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

logging.getLogger("pgmpy").setLevel(logging.ERROR)

# ============================================================
# Adult configuration
# ============================================================

SEED = 42

# Requires a working XGBoost installation, including libomp on macOS.
REGRESSOR = "xgboost"

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERICAL_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


# ============================================================
# Prepare numerical input for your model
# ============================================================


def prepare_forest_data(
    train_df: pd.DataFrame,
    target_col: str,
    model_data_dir: Path,
) -> tuple[dict[str, list[str]], list[int], list[int]]:
    """Encode training data, including the target, for joint generation."""
    required_columns = [
        *CATEGORICAL_COLUMNS,
        *NUMERICAL_COLUMNS,
        target_col,
    ]

    missing_columns = [
        column for column in required_columns if column not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}. "
            f"Available columns: {train_df.columns.tolist()}"
        )

    encoded_df = train_df.copy()
    category_maps = {}

    # Include the target so the model learns its relationship to features.
    discrete_columns = [*CATEGORICAL_COLUMNS, target_col]

    for column in discrete_columns:
        values = (
            train_df[column]
            .astype("string")
            .str.strip()
            .replace({"?": pd.NA, "": pd.NA})
            .fillna("Unknown")
        )

        categories = sorted(values.unique().tolist())
        mapping = {category: index for index, category in enumerate(categories)}

        encoded_df[column] = values.map(mapping).astype(float)
        category_maps[column] = categories

    # Fit missing-value replacement using training data only.
    for column in NUMERICAL_COLUMNS:
        values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        median = values.median()

        if pd.isna(median):
            raise ValueError(
                f"Numerical column '{column}' has no valid training values."
            )

        encoded_df[column] = values.fillna(median).astype(float)

    encoded_df = encoded_df.astype(float)

    if not np.isfinite(encoded_df.to_numpy()).all():
        raise ValueError("Encoded training data contains NaN or infinity.")

    model_data_dir.mkdir(parents=True, exist_ok=True)

    # model reads x_train.csv. Here it includes the target column.
    encoded_df.to_csv(
        model_data_dir / "x_train.csv",
        index=False,
    )

    with (model_data_dir / "category_maps.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(category_maps, file, indent=2)

    categorical_indexes = [
        encoded_df.columns.get_loc(column) for column in discrete_columns
    ]

    integer_indexes = [
        encoded_df.columns.get_loc(column) for column in NUMERICAL_COLUMNS
    ]

    return category_maps, categorical_indexes, integer_indexes


# ============================================================
# Restore generated data to Adult format
# ============================================================


def decode_forest_output(
    generated_df: pd.DataFrame,
    train_df: pd.DataFrame,
    category_maps: dict[str, list[str]],
) -> pd.DataFrame:
    """Restore category labels and integer-valued Adult columns."""
    synthetic_df = generated_df.copy()

    missing_columns = [
        column for column in train_df.columns if column not in synthetic_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Generated data is missing columns: {missing_columns}")

    for column, categories in category_maps.items():
        values = pd.to_numeric(
            synthetic_df[column],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Generated categorical column '{column}' is not finite.")

        codes = np.clip(
            np.rint(values),
            0,
            len(categories) - 1,
        ).astype(int)

        synthetic_df[column] = np.asarray(categories)[codes]

    for column in NUMERICAL_COLUMNS:
        values = pd.to_numeric(
            synthetic_df[column],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Generated numerical column '{column}' is not finite.")

        # The model already reverses its scaling and clips to training bounds.
        synthetic_df[column] = np.rint(values).astype(np.int64)

    return synthetic_df.loc[:, train_df.columns]


# ============================================================
# System summary
# ============================================================


def print_system_summary() -> None:
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
        dataset_name="adult",
        model_name="forest_diffusion",
        categorical_cols=CATEGORICAL_COLUMNS,
        continuous_cols=NUMERICAL_COLUMNS,
        target_col_raw="class",
        constraints={},
    )

    print_system_summary()

    print("\nSTEP 1 — Preprocess and split Adult")

    train_df, test_df, target_col, paths = preprocess_and_split(config)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")
    print(f"Target column: {target_col}")

    # Keep model-specific encoded files separate from shared dataset splits.
    model_data_dir = Path(paths["split_dir"]) / "forest_diffusion_joint"

    # Keep the model's intermediate CSVs separate from final benchmark outputs.
    model_output_dir = Path(paths["synthetic_dir"]) / "forest_diffusion_joint"

    print("\nSTEP 2 — Encode features and target")

    category_maps, categorical_indexes, integer_indexes = prepare_forest_data(
        train_df=train_df,
        target_col=target_col,
        model_data_dir=model_data_dir,
    )

    print(f"Categorical indexes: {categorical_indexes}")
    print(f"Integer indexes: {integer_indexes}")

    # Fail clearly if XGBoost cannot load instead of allowing the model's
    # broad exception handler to silently switch to Random Forest.
    if REGRESSOR == "xgboost":
        import xgboost

        print(f"XGBoost version: {xgboost.__version__}")

    print("\nSTEP 3 — Train and generate Forest Diffusion data")

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

    model.category_maps = category_maps

    training_start = perf_counter()

    model.category_maps = category_maps

    # train() method also generates and saves synthetic rows.
    model.train(
        dataset_dir=str(model_data_dir),
        synthetic_dir=str(model_output_dir),
    )

    model.category_maps = category_maps

    training_seconds = perf_counter() - training_start

    print("\nSTEP 4 — Decode and save synthetic data")

    # Since class was included in x_train.csv, it is generated jointly here.
    # Ignore the dummy y_synth.csv written by your model.
    generated_df = pd.read_csv(model_output_dir / "x_synth.csv")

    if len(generated_df) != len(train_df):
        raise ValueError(
            f"Expected {len(train_df)} synthetic rows, "
            f"but received {len(generated_df)}."
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

    print("\nSTEP 5 — Evaluate synthetic data")

    for column in config.categorical_cols:
        print(f"\n{column}")
        print("Real:", train_df[column].dropna().unique()[:10])
        print("Synthetic:", synthetic_df[column].dropna().unique()[:10])
        print(
            "Types:",
            train_df[column].dtype,
            synthetic_df[column].dtype,
        )

        test_sample = model.sample(1000, seed=42)

    for column in config.categorical_cols:
        real_categories = set(train_df[column].astype(str).str.strip().unique())
        synthetic_categories = set(test_sample[column].astype(str).str.strip().unique())

        matched_categories = real_categories & synthetic_categories

        print(f"\n{column}")
        print("Real categories:", sorted(real_categories))
        print("Synthetic categories:", sorted(synthetic_categories))
        print("Matched categories:", sorted(matched_categories))

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
    print(f"Model training and generation: {training_seconds:.2f} seconds")
    print(f"Complete pipeline: {elapsed_seconds:.2f} seconds")
    print(f"Synthetic directory: {paths['synthetic_dir']}")
    print(f"Results directory: {paths['results_dir']}")


if __name__ == "__main__":
    main()
