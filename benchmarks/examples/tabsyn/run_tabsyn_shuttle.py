import json
import logging
import os
import platform
import sys
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

logging.getLogger("pgmpy").setLevel(logging.ERROR)


# ============================================================
# PROJECT PATH SETUP
# ============================================================

# Current script location:
# Katabatic/benchmarks/examples/tabsyn/run_tabsyn_shuttle.py
#
# Moving up three directories reaches:
# Katabatic/
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

BENCHMARKS_DIR = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)


# These imports must come after the project path setup.
from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.tabsyn.models import TabSyn  # noqa: E402

# Run in CPU mode.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

start_time = perf_counter()


# ============================================================
# RUNTIME SUMMARY
# ============================================================


def get_runtime_summary(
    time_diff,
    start_time,
    end_time,
    model_name,
    dataset_name,
) -> None:
    """Print a formatted runtime summary report."""

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
# SYSTEM INFORMATION
# ============================================================


def get_system_run_details() -> None:
    """Print system, processor, GPU, and RAM information."""

    results = platform.uname()
    ram = psutil.virtual_memory()

    gpu_info = "No GPU has been detected."

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
# PREPARE FILES REQUIRED BY TABSyn utils.py
# ============================================================


def prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    continuous_cols,
):
    """Create NumPy files required by the TabSyn utility code."""

    # Continuous features.
    x_num_train = train_df[continuous_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    x_num_test = test_df[continuous_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Fill missing values using training medians.
    medians = x_num_train.median()

    x_num_train = x_num_train.fillna(medians)
    x_num_test = x_num_test.fillna(medians)

    # Shuttle has no categorical feature columns.
    x_cat_train = np.empty(
        (len(train_df), 0),
        dtype=str,
    )

    x_cat_test = np.empty(
        (len(test_df), 0),
        dtype=str,
    )

    # Target.
    y_train = train_df[target_col].fillna("Unknown").astype(str).to_numpy()

    y_test = test_df[target_col].fillna("Unknown").astype(str).to_numpy()

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_train.npy",
        ),
        x_num_train.to_numpy(dtype=np.float32),
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_test.npy",
        ),
        x_num_test.to_numpy(dtype=np.float32),
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_cat_train.npy",
        ),
        x_cat_train,
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_cat_test.npy",
        ),
        x_cat_test,
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
        "n_classes": int(train_df[target_col].nunique()),
    }

    with open(
        os.path.join(
            paths["split_dir"],
            "info.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            info,
            file,
            indent=4,
        )


# ============================================================
# CONVERT TABSyn OUTPUT TO SHUTTLE DATASET FORMAT
# ============================================================


def convert_tabsyn_output(
    sampled_df,
    train_df,
    continuous_cols,
    target_col,
):
    """Convert TabSyn output back to the Shuttle dataset schema."""

    synthetic_df = pd.DataFrame(index=sampled_df.index)

    # --------------------------------------------------------
    # Continuous features
    # --------------------------------------------------------

    for index, column in enumerate(continuous_cols):
        generated_values = pd.to_numeric(
            sampled_df[f"num_{index}"],
            errors="coerce",
        ).fillna(0)

        real_values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        mean = real_values.mean()
        std = real_values.std(ddof=0)

        if pd.isna(std) or std == 0:
            std = 1.0

        values = generated_values * std + mean

        # Shuttle values are integer based.
        values = values.round()

        # Keep synthetic values inside the real data range.
        values = values.clip(
            lower=real_values.min(),
            upper=real_values.max(),
        )

        synthetic_df[column] = values.astype(int)

    # --------------------------------------------------------
    # Target
    #
    # cat_0 represents the target.
    # --------------------------------------------------------

    target_categories = np.unique(train_df[target_col].fillna("Unknown").astype(str))

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

    synthetic_df[target_col] = [target_categories[index] for index in target_indices]

    # Restore exact Shuttle column order.
    synthetic_df = synthetic_df[train_df.columns]

    return synthetic_df


# ============================================================
# SHUTTLE DATASET CONFIGURATION
# ============================================================


config = RunConfig(
    dataset_name="shuttle",
    model_name="tabsyn",
    categorical_cols=[],
    continuous_cols=[
        "time",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
    ],
    target_col_raw="class",
    constraints={},
)


# ============================================================
# PREPROCESS AND SPLIT
# ============================================================


train_df, test_df, target_col, paths = preprocess_and_split(config)


# Create NumPy files expected by TabSyn utils.py.
prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    config.continuous_cols,
)


# ============================================================
# STEP 3 — TRAIN TABSyn
# ============================================================


print("\n" + "=" * 60)
print("STEP 3 — Train TabSyn")
print("=" * 60)


model = TabSyn(
    d_token=16,
    decoder_epochs=50,
    decoder_batch_size=512,
    diffusion_epochs=300,
    diffusion_batch_size=512,
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
        "categorical_cols": config.categorical_cols,
        "continuous_cols": config.continuous_cols,
        "target_col": target_col,
    },
)


print("\nTabSyn training complete.")


# ============================================================
# STEP 4 — GENERATE SYNTHETIC DATA
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
    config.continuous_cols,
    target_col,
)


synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)


# ============================================================
# EVALUATE
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
# RUNTIME AND SYSTEM SUMMARY
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
