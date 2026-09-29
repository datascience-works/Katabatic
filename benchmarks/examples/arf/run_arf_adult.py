import logging
import os
import platform
import sys
import warnings
from time import perf_counter

import psutil

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.arf.models import ARFModel

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

start_time = perf_counter()


def get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    model_name,
    dataset_name,
) -> None:
    """Print the runtime summary for the model evaluation."""
    print("=" * 70)
    print("Evaluation Runtime Report")
    print("=" * 70)
    print("Start time:", start_time)
    print("End time:", end_time)
    print(
        f"{model_name} has taken {time_diff:.2f} seconds "
        f"to run the {dataset_name} dataset."
    )


def get_system_run_details() -> None:
    """Print the hardware used for the evaluation."""
    results = platform.uname()
    ram = psutil.virtual_memory()

    print("=" * 70)
    print("Computation Hardware Summary")
    print("=" * 70)
    print(f"  System:     {results.system}")
    print(f"  Node:       {results.node}")
    print(f"  Release:    {results.release}")
    print(f"  Version:    {results.version}")
    print(f"  Processor:  {results.processor}")
    print(f"  Total RAM:  {round(ram.total / 1e9, 4)} GB")
    print(f"  Free RAM:   {round(ram.available / 1e9, 4)} GB")
    print(f"  Used RAM:   {round(ram.used / 1e9, 4)} GB")
    print("=" * 70)


config = RunConfig(
    dataset_name="adult",
    model_name="arf",
    categorical_cols=[
        "workclass",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ],
    continuous_cols=[
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    target_col_raw="class",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train ARF")
print("=" * 60)

model = ARFModel(
    num_trees=30,
    max_iters=10,
    delta=0.0,
    min_node_size=5,
    seed=42,
    leaf_thresh=0.5,
)

model.train(
    paths["split_dir"],
    n_synth=len(train_df),
)

print("\nARF training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(len(train_df))

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

end_time = perf_counter()
time_diff = end_time - start_time

get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    config.model_name,
    config.dataset_name,
)

get_system_run_details()
