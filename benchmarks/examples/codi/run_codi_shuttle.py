import logging
import os
import platform
import sys
import warnings
from time import perf_counter

import pandas as pd
import psutil

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import (
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.codi.models import CODI  # noqa: E402

# run in cpu mode(if GPU is limited)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
start_time = perf_counter()


# ➕ Adding in system and run duration summary
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

    Args:
        time_diff (timedelta): Total elapsed time of the run.
        start_time (datetime): Start timestamp of the run.
        end_time (datetime): End timestamp of the run.
        model_name (str): Name of the model used.
        dataset_name (str): Name of the dataset used.
    """
    print("======================================================================")
    print("⏰ Evaluation Runtime Report 🧾")
    print("======================================================================")
    print("Start time:", start_time)
    print("End time:", end_time)
    print(
        model_name
        + " has taken "
        + str(time_diff)
        + " seconds to run the adult "
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
        gpu_info = "No GPU or Tensorflow install has been detected."

    print("======================================================================")
    print("💻 Computation Hardware Summary 🧾")
    print("======================================================================")
    print(f"  🖥️  System:     {results.system}")
    print(f"  🏠  Node:       {results.node}")
    print(f"  📦  Release:    {results.release}")
    print(f"  🔢  Version:    {results.version}")
    print(f"  🔧  Processor:  {results.processor}")
    print(f"  🎮  GPU:        {gpu_info}")
    print(f"  📟  Total RAM:  {round(ram.total / 1e9, 4)} GB")
    print(f"  💾  Free RAM:   {round(ram.available / 1e9, 4)} GB")
    print(f"  ⚡  Used RAM:   {round(ram.used / 1e9, 4)} GB")
    print("======================================================================")


config = RunConfig(
    dataset_name="shuttle",
    model_name="codi",
    categorical_cols=[],
    continuous_cols=["time", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    target_col_raw="class",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train CODI")
print("=" * 60)
model = CODI(n_steps=50, epochs=100, batch_size=256)
model.train(
    paths["split_dir"],
    paths["synthetic_dir"], 
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nCoDi training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = pd.DataFrame(model.sample(len(train_df)), columns=train_df.columns)
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

# ➕ Adding in system and run duration summary
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
