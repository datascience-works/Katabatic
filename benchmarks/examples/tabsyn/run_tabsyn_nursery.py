import platform
import psutil
import sys
import os
import json
import numpy as np
from time import perf_counter
import warnings
import pandas as pd
import logging
logging.getLogger("pgmpy").setLevel(logging.ERROR)

# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, "benchmarks")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

from runner import (
    RunConfig,
    preprocess_and_split,
    save_synthetic,
    evaluate,
)

from katabatic.models.tabsyn.models import TabSyn

# Run in CPU mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

start_time = perf_counter()

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

# ============================================================
# System information
# ============================================================

def get_system_run_details() -> None:

    results = platform.uname()
    ram = psutil.virtual_memory()

    gpu_info = "No GPU has been detected."

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            gpu_info = "Apple Silicon GPU (MPS available)"

        else:
            gpu_info = "CPU"

    except Exception:
        gpu_info = "No GPU or PyTorch GPU support detected."

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
# ============================================================
# Prepare files required by TabSyn utils.py
# ============================================================

def prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    categorical_cols,
):

    # Nursery has no continuous columns.
    X_num_train = np.empty(
        (len(train_df), 0),
        dtype=np.float32,
    )

    X_num_test = np.empty(
        (len(test_df), 0),
        dtype=np.float32,
    )

    # Categorical feature columns.
    X_cat_train = (
        train_df[categorical_cols]
        .fillna("Unknown")
        .astype(str)
    )

    X_cat_test = (
        test_df[categorical_cols]
        .fillna("Unknown")
        .astype(str)
    )

    # Target.
    y_train = (
        train_df[target_col]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    y_test = (
        test_df[target_col]
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_train.npy",
        ),
        X_num_train,
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_test.npy",
        ),
        X_num_test,
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_cat_train.npy",
        ),
        X_cat_train.to_numpy(dtype=str),
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_cat_test.npy",
        ),
        X_cat_test.to_numpy(dtype=str),
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "y_train.npy",
        ),
        y_train,
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "y_test.npy",
        ),
        y_test,
    )

    info = {
        "task_type": "multiclass",
        "n_classes": int(
            train_df[target_col].nunique()
        ),
    }

    with open(
        os.path.join(
            paths["split_dir"],
            "info.json",
        ),
        "w",
    ) as file:

        json.dump(
            info,
            file,
            indent=4,
        )

# ============================================================
# Convert TabSyn output to Nursery dataset format
# ============================================================

def convert_tabsyn_output(
    sampled_df,
    train_df,
    categorical_cols,
    target_col,
):

    synthetic_df = pd.DataFrame(
        index=sampled_df.index
    )

    # --------------------------------------------------------
    # Target
    #
    # cat_0 = target
    # --------------------------------------------------------

    target_categories = np.unique(
        train_df[target_col]
        .fillna("Unknown")
        .astype(str)
    )

    target_indices = (
        pd.to_numeric(
            sampled_df["cat_0"],
            errors="coerce",
        )
        .fillna(0)
        .round()
        .astype(int)
        .clip(
            lower=0,
            upper=len(target_categories) - 1,
        )
    )

    synthetic_df[target_col] = [
        target_categories[index]
        for index in target_indices
    ]

    # --------------------------------------------------------
    # Categorical features
    #
    # cat_1 = column 0
    # cat_2 = column 1
    # ...
    # cat_8 = column 7
    # --------------------------------------------------------

    for i, col in enumerate(
        categorical_cols,
        start=1,
    ):

        categories = np.unique(
            train_df[col]
            .fillna("Unknown")
            .astype(str)
        )

        indices = (
            pd.to_numeric(
                sampled_df[f"cat_{i}"],
                errors="coerce",
            )
            .fillna(0)
            .round()
            .astype(int)
            .clip(
                lower=0,
                upper=len(categories) - 1,
            )
        )

        synthetic_df[col] = [
            categories[index]
            for index in indices
        ]

    # Restore exact Nursery column order.
    synthetic_df = synthetic_df[
        train_df.columns
    ]

    return synthetic_df

# ============================================================
# Nursery dataset configuration
# ============================================================

config = RunConfig(
    dataset_name="nursery",
    model_name="tabsyn",
    categorical_cols=["0","1","2","3","4","5","6","7",],
    continuous_cols=[],
    target_col_raw="8",
    constraints={},
)

train_df, test_df, target_col, paths = (
    preprocess_and_split(config)
)

# Create NumPy files expected by TabSyn utils.py.
prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    config.categorical_cols,
)

# ============================================================
# STEP 3 — Train TabSyn
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 — Train TabSyn")
print("=" * 60)

model = TabSyn(
    d_token=16,

    decoder_epochs=50,
    decoder_batch_size=256,

    diffusion_epochs=300,
    diffusion_batch_size=256,

    diffusion_steps=50,

    lr=1e-3,
    weight_decay=0.0,

    patience=20,
    seed=42,

    device="cpu",
)

model.train(
    paths["split_dir"],

    save_dir=os.path.join(
        paths["results_dir"],
        "model_files",
    ),

    synthetic_dir=paths["synthetic_dir"],

    extra_info={
        "categorical_cols":
            config.categorical_cols,

        "continuous_cols":
            config.continuous_cols,

        "target_col":
            target_col,
    },
)

print("\nTabSyn training complete.")

# ============================================================
# STEP 4 — Generate synthetic data
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)

sampled_df = model.sample(
    n_samples=len(train_df),
    return_df=True,
)

synthetic_df = convert_tabsyn_output(
    sampled_df,
    train_df,
    config.categorical_cols,
    target_col,
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

# ============================================================
# Runtime and system summary
# ============================================================

end_time = perf_counter()

time_diff = (
    end_time
    - start_time
)
get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    config.model_name,
    config.dataset_name,
)
get_system_run_details()