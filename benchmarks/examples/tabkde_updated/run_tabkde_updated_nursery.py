import logging
import os
import platform
import random
import sys
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
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

from katabatic.models.tabkde_updated import TabKDEModel  # noqa: E402

# Run in CPU mode if GPU is limited
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
start_time = perf_counter()


class TabKDEEvaluationAdapter(TabKDEModel):
    """
    Make TabKDE compatible with the Katabatic evaluation pipeline.

    The original TabKDE sample() method may return a tuple,
    while the evaluation pipeline expects a pandas DataFrame.

    The stability evaluator also passes a seed argument.
    """

    def sample(self, n, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        result = super().sample(n)

        if isinstance(result, tuple):
            return result[0]

        return result


# Adding system and run duration summary
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

    gpu_info = "No GPU has been detected."

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_info = details.get("device_name", "Unknown")

    except Exception:
        gpu_info = "No GPU or TensorFlow install has been detected."

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
    model_name="tabkde_updated",
    categorical_cols=["0", "1", "2", "3", "4", "5", "6", "7"],
    continuous_cols=[],
    target_col_raw="8",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 - Train TabKDE Updated - Nursery")
print("=" * 60)

model = TabKDEEvaluationAdapter()
model.fit(train_df)

print("\nTabKDE Updated training complete.")

print("\n" + "=" * 60)
print("STEP 4 - Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(len(train_df))

if not isinstance(synthetic_df, pd.DataFrame):
    synthetic_df = pd.DataFrame(
        synthetic_df,
        columns=train_df.columns,
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

# Adding system and run duration summary
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
