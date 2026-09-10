import logging
import os
import platform
import sys
import warnings
from time import perf_counter

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

from katabatic.models.model_hyper_parameters import MODELS

# Enter the model and dataset you want to run

print("Models available are:")
print(list(MODELS), " or select 'ALL' to run all models.")
print("Select the model you wish to run.")

model_chosen = input().upper()

if model_chosen == "ALL":
    model_chosen = list(MODELS)
elif model_chosen not in list(MODELS):
    sys.exit("Error: Only the outlined choices can be entered!")
else:
    model_chosen = [model_chosen]

all_model_results = {}

# run in cpu mode( "CUDA_VISIBLE_DEVICES" = "-1" or 0 for GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


for m in model_chosen:
    config = RunConfig(
        dataset_name="shuttle",
        model_name=m,
        #target_col_raw="6",
    )

    train_df, test_df, target_col, paths = preprocess_and_split(config)

    print("\n" + "=" * 60)
    print("STEP 3 — Train ", m)
    print("=" * 60)

    model_config = MODELS[m]
    model_hp = model_config["class"](**model_config["params"])

    model = model_hp  # CTGANModel(epochs=100, batch_size=512, seed=42)
    model.train(
        paths["split_dir"],
        paths["synthetic_dir"],
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
    print("\n", m, " training complete.")

    print("\n" + "=" * 60)
    print("STEP 4 — Generate synthetic data")
    print("=" * 60)
    synthetic_df = model.sample(len(train_df))
    synthetic_df = save_synthetic(
        synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
    )

    composite_score = evaluate(
        model=model,
        config=config,
        train_df=train_df,
        synthetic_df=synthetic_df,
        target_col=target_col,
        paths=paths,
        test_df=test_df,
    ).composite_score
    all_model_results[m] = composite_score

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

# Recommend the model to used based on the highest composite score

print("\n======= RECOMMENDED MODEL TO USE =======")
best_score = 0
best_model = ""
for m_results in all_model_results:
    if best_score <= all_model_results[m_results]:
        best_score = all_model_results[m_results]
        best_model = m_results
text_final = """
From the data selected and hyperparameters chosen with in the file model_hyper_parameters.py
the analysis has shown the following model best suits your data for the 6 metrics:
"""

print(text_final)
print(f"{best_model} with a score of {best_score:.4f}")

print("\n======= MODEL SCORE SUMMARY =======")

sorted_results = sorted(
    all_model_results.items(),
    key=lambda x: x[1],
    reverse=True
)

model_width = max(
    len("Model"),
    max(len(str(model)) for model in all_model_results)
)

print(f"{'Rank':<5}{'Model':<{model_width + 5}}{'Score':>5}")
print("-" * (model_width + 19))

for rank, (model, score) in enumerate(sorted_results, start=1):
    marker = " <-- Recommended" if model == best_model else ""
    print(
        f"{rank:<5}"
        f"{model:<{model_width + 5}}"
        f"{score:>5.4f}"
        f"{marker}"
    )
