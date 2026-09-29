import logging
import os
import platform
import sys
import warnings
from time import perf_counter

import pandas as pd
import psutil
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabula.models import TABULA  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
start_time = perf_counter()


def get_runtime_summary(time_diff, start_time, end_time, model_name, dataset_name) -> None:
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


# Device detection — supports CUDA, Apple Silicon (MPS), and CPU fallback
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Paper: Zhao et al. (2023) "Tabula: Harnessing Language Models for Tabular Data Synthesis"
# arXiv:2310.12746. Car Evaluation dataset, small dataset setting.
# Paper hyperparameters: epochs=100 for small datasets, distilgpt2 with random init, k=16.
config = RunConfig(
    dataset_name="car",
    model_name="tabula",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample for validation on CPU/MPS — cap at 50 rows per class (200 total for 4 classes).
# Rows are short (all categorical) so text sequences are compact; runtime is ~15 min.
# Remove this block and run the full dataset on GPU to reproduce paper results.
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 50), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(
    os.path.join(paths["split_dir"], "x_train.csv"), index=False
)
_sub[[target_col]].to_csv(
    os.path.join(paths["split_dir"], "y_train.csv"), index=False
)

print("\n" + "=" * 60)
print("STEP 3 — Train TabuLa")
print("=" * 60)
model = TABULA(
    categorical_columns=config.categorical_cols,
    epochs=100,  # paper: 100 epochs for small datasets (Zhao et al., 2023, Table 2)
)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    n_samples=1000,
    k=16,           # paper: k=16 beam search candidates (Zhao et al., 2023, Section 4.2)
    max_length=128,
    max_rounds=100,
)
print("\nTabuLa training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
x_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "x_synth.csv"))
y_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "y_synth.csv"))
synthetic_df = pd.concat([x_synth, y_synth], axis=1)
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)

end_time = perf_counter()
time_diff = end_time - start_time
get_runtime_summary(time_diff, start_time, end_time, config.model_name, config.dataset_name)
get_system_run_details()