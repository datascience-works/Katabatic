import os
import platform
import sys
from time import perf_counter

import psutil

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ),
)

from runner import (
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.arf.models import ARF  # noqa: E402

SEED = 42
MAX_ITERS = 10
start_time = perf_counter()


def get_runtime_summary(
    duration: float,
    model_name: str,
    dataset_name: str,
) -> None:
    """Print total evaluation runtime."""

    print("=" * 70)
    print("Evaluation Runtime Report")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(
        f"Total duration: "
        f"{duration:.2f} seconds"
    )


def get_system_run_details() -> None:
    """Print hardware information."""

    results = platform.uname()
    ram = psutil.virtual_memory()

    print("=" * 70)
    print("Computation Hardware Summary")
    print("=" * 70)
    print(f"System: {results.system}")
    print(f"Node: {results.node}")
    print(f"Release: {results.release}")
    print(f"Processor: {results.processor}")
    print("GPU: Not used by ARF")
    print(
        f"Total RAM: "
        f"{ram.total / 1e9:.4f} GB"
    )
    print(
        f"Available RAM: "
        f"{ram.available / 1e9:.4f} GB"
    )
    print(
        f"Used RAM: "
        f"{ram.used / 1e9:.4f} GB"
    )
    print("=" * 70)


config = RunConfig(
    dataset_name="adult",
    model_name="arf",
    categorical_cols=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    continuous_cols=[
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    target_col_raw="class",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "educational-num": (1, 16),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    },
)


train_df, test_df, target_col, paths = (
    preprocess_and_split(config)
)


print("\n" + "=" * 60)
print("STEP 3 - Train ARF")
print("=" * 60)

model = ARF(
    max_iters=MAX_ITERS,
    seed=SEED,
)

model.fit(
    train_df.drop(columns=[target_col]),
    train_df[target_col],
    categorical_cols=config.categorical_cols,
    seed=SEED,
)

print(
    "\nARF training and FORDE "
    "density estimation complete."
)


print("\n" + "=" * 60)
print("STEP 4 - Generate synthetic data with FORGE")
print("=" * 60)

synthetic_df = model.sample(
    len(train_df),
    seed=SEED,
)

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)


evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)


duration = perf_counter() - start_time

get_runtime_summary(
    duration,
    config.model_name,
    config.dataset_name,
)

get_system_run_details()