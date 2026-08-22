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
    dataset_name="bank_marketing",
    model_name="tabkde_updated",
    categorical_cols=[
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ],
    continuous_cols=[
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ],
    target_col_raw="y",
    constraints={
        "age": (18, 95),
        "balance": (-8020, None),
        "day": (1, 31),
        "duration": (0, None),
        "campaign": (1, None),
        "pdays": (-1, None),
        "previous": (0, None),
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
print("STEP 3 — Train TabKDE Updated - Bank Marketing")
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