import platform
import psutil
import sys
import os
from time import perf_counter
import warnings
import pandas as pd
import logging
import random
import numpy as np

logging.getLogger("pgmpy").setLevel(logging.ERROR)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from runner import (
    RunConfig,
    preprocess_and_split,
    save_synthetic,
    evaluate,
)

from katabatic.models.tabkde_updated import TabKDEModel


# Run in CPU mode if GPU is limited
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

start_time = perf_counter()


# ============================================================
# TabKDE compatibility adapter
# ============================================================
class TabKDEEvaluationAdapter(TabKDEModel):
    """
    Small adapter to make TabKDE compatible with the
    Katabatic evaluation pipeline.

    The original TabKDE sample() method may return a tuple,
    while the evaluation pipeline expects a pandas DataFrame.

    The stability evaluator also passes a seed argument.
    """

    def sample(self, n, seed=None):

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        result = super().sample(n)

        # TabKDE may return (synthetic_df, additional_output)
        if isinstance(result, tuple):
            return result[0]

        return result


# ============================================================
# Runtime summary
# ============================================================
def get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    model_name,
    dataset_name,
) -> None:
    """
    Print a formatted runtime summary report.

    Args:
        time_diff: Total elapsed time of the run.
        start_time: Start time of the run.
        end_time: End time of the run.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
    """

    print(
        "======================================================================"
    )
    print("⏰ Evaluation Runtime Report 🧾")
    print(
        "======================================================================"
    )

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


# ============================================================
# System information
# ============================================================
def get_system_run_details() -> None:
    """
    Print a summary of the hardware used to run the evaluation.
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

    print(
        "======================================================================"
    )
    print("💻 Computation Hardware Summary 🧾")
    print(
        "======================================================================"
    )

    print(f"  🖥️  System:     {results.system}")
    print(f"  🏠  Node:       {results.node}")
    print(f"  📦  Release:    {results.release}")
    print(f"  🔢  Version:    {results.version}")
    print(f"  🔧  Processor:  {results.processor}")
    print(f"  🎮  GPU:        {gpu_info}")
    print(f"  📟  Total RAM:  {round(ram.total / 1e9, 4)} GB")
    print(f"  💾  Free RAM:   {round(ram.available / 1e9, 4)} GB")
    print(f"  ⚡  Used RAM:   {round(ram.used / 1e9, 4)} GB")

    print(
        "======================================================================"
    )


# ============================================================
# Configuration
# ============================================================

config = RunConfig(
    dataset_name="creditcard",
    model_name="tabkde_updated",
    categorical_cols=[],
    continuous_cols=[
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ],
    target_col_raw="Class",
    constraints={
        "Time": (0.0, 172792.0),
        "V1": (-56.41, 2.45),
        "V2": (-72.72, 22.06),
        "V3": (-48.33, 9.38),
        "V4": (-5.68, 16.88),
        "V5": (-113.74, 34.80),
        "V6": (-26.16, 73.30),
        "V7": (-43.56, 120.59),
        "V8": (-73.22, 20.01),
        "V9": (-13.43, 15.60),
        "V10": (-24.59, 23.75),
        "V11": (-4.80, 12.02),
        "V12": (-18.68, 7.85),
        "V13": (-5.79, 7.13),
        "V14": (-19.21, 10.53),
        "V15": (-4.50, 8.88),
        "V16": (-14.13, 17.32),
        "V17": (-25.16, 9.25),
        "V18": (-9.50, 5.04),
        "V19": (-7.21, 5.59),
        "V20": (-54.50, 39.42),
        "V21": (-34.83, 27.20),
        "V22": (-10.93, 10.50),
        "V23": (-44.81, 22.53),
        "V24": (-2.84, 4.58),
        "V25": (-10.30, 7.52),
        "V26": (-2.60, 3.52),
        "V27": (-22.57, 31.61),
        "V28": (-15.43, 33.85),
        "Amount": (0.0, 25691.16),
    },
)
# ============================================================
# STEP 1 & STEP 2 — Preprocess and split
# ============================================================
train_df, test_df, target_col, paths = preprocess_and_split(config)


# ============================================================
# STEP 3 — Train TabKDE Updated
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 — Train TabKDE Updated - Credit Card")
print("=" * 60)

model = TabKDEEvaluationAdapter()

model.fit(train_df)

print("\nTabKDE Updated training complete.")


# ============================================================
# STEP 4 — Generate synthetic data
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(len(train_df))

# Make sure output is a DataFrame
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


# ============================================================
# STEP 5 — Evaluate synthetic data
# ============================================================
evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)


# ============================================================
# Runtime and hardware summary
# ============================================================
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