import os
import platform
import sys
from time import perf_counter

import psutil

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.gmm import GMMModel  # noqa: E402

start_time = perf_counter()


def get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    model_name,
    dataset_name,
) -> None:
    """
    Print a formatted runtime summary report.

    The report includes the start time, end time, and total duration for a
    given model and a specific dataset.
    """
    print("======================================================================")
    print("Evaluation Runtime Report")
    print("======================================================================")
    print("Start time:", start_time)
    print("End time:", end_time)
    print(
        model_name
        + " has taken "
        + str(time_diff)
        + " seconds to run the "
        + dataset_name
        + " dataset."
    )


def get_system_run_details() -> None:
    """
    Print a summary of the hardware used to run the evaluations.

    This report outlines the OS, CPU, GPU details, and RAM states.
    It is compatible with any device or OS.
    """
    results = platform.uname()
    ram = psutil.virtual_memory()

    gpu_info = "Not used by GMM."

    print("======================================================================")
    print("Computation Hardware Summary")
    print("======================================================================")
    print(f"  System:     {results.system}")
    print(f"  Node:       {results.node}")
    print(f"  Release:    {results.release}")
    print(f"  Version:    {results.version}")
    print(f"  Processor:  {results.processor}")
    print(f"  GPU:        {gpu_info}")
    print(f"  Total RAM:  {round(ram.total / 1e9, 4)} GB")
    print(f"  Free RAM:   {round(ram.available / 1e9, 4)} GB")
    print(f"  Used RAM:   {round(ram.used / 1e9, 4)} GB")
    print("======================================================================")


config = RunConfig(
    dataset_name="nursery",
    model_name="gmm",
    categorical_cols=[
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ],
    continuous_cols=[],
    target_col_raw="8",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 - Train GMM - Nursery")
print("=" * 60)

model = GMMModel(
    target_col=target_col,
    n_components=4,
    covariance_type="full",
    random_state=42,
)

model.fit(train_df)

print("\nGMM training complete.")

print("\n" + "=" * 60)
print("STEP 4 - Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(
    len(train_df),
    seed=42,
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
