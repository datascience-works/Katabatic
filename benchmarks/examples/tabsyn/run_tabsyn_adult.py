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


from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.models.tabsyn.models import TabSyn  # noqa: E402

logging.getLogger("pgmpy").setLevel(logging.ERROR)

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
# Prepare files required by TabSyn utils.py
# ============================================================


def prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    continuous_cols,
    categorical_cols,
):

    # Numerical columns
    X_num_train = train_df[continuous_cols].apply(pd.to_numeric, errors="coerce")

    X_num_test = test_df[continuous_cols].apply(pd.to_numeric, errors="coerce")

    medians = X_num_train.median()

    X_num_train = X_num_train.fillna(medians)
    X_num_test = X_num_test.fillna(medians)

    # Categorical columns
    X_cat_train = train_df[categorical_cols].fillna("Unknown").astype(str)

    X_cat_test = test_df[categorical_cols].fillna("Unknown").astype(str)

    # Target
    y_train = train_df[target_col].fillna("Unknown").astype(str).to_numpy()

    y_test = test_df[target_col].fillna("Unknown").astype(str).to_numpy()

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_train.npy",
        ),
        X_num_train.to_numpy(dtype=np.float32),
    )

    np.save(
        os.path.join(
            paths["split_dir"],
            "X_num_test.npy",
        ),
        X_num_test.to_numpy(dtype=np.float32),
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
        "task_type": "binclass",
        "n_classes": int(train_df[target_col].nunique()),
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
# Convert TabSyn sample into Adult dataset format
# ============================================================


def convert_tabsyn_output(
    sampled_df,
    train_df,
    continuous_cols,
    categorical_cols,
    target_col,
):

    synthetic_df = pd.DataFrame(index=sampled_df.index)

    # --------------------------------------------------------
    # Continuous columns
    # --------------------------------------------------------

    for i, col in enumerate(continuous_cols):
        generated_values = pd.to_numeric(
            sampled_df[f"num_{i}"],
            errors="coerce",
        ).fillna(0)

        real_values = pd.to_numeric(
            train_df[col],
            errors="coerce",
        )

        mean = real_values.mean()
        std = real_values.std(ddof=0)

        if pd.isna(std) or std == 0:
            std = 1.0

        # TabSyn output is on standardised scale.
        values = generated_values * std + mean

        # Adult numeric columns are integer based.
        values = values.round()

        # Keep generated values inside the real range.
        values = values.clip(
            lower=real_values.min(),
            upper=real_values.max(),
        )

        synthetic_df[col] = values.astype(int)

    # --------------------------------------------------------
    # Target column
    # cat_0 represents the target
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
    # --------------------------------------------------------
    # Categorical feature columns
    #
    # cat_0 = target
    # cat_1 = workclass
    # cat_2 = education
    # ...
    # --------------------------------------------------------

    for i, col in enumerate(
        categorical_cols,
        start=1,
    ):
        categories = np.unique(train_df[col].fillna("Unknown").astype(str))

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

        synthetic_df[col] = [categories[index] for index in indices]

    # Restore exact Adult column order.
    synthetic_df = synthetic_df[train_df.columns]

    return synthetic_df


# ============================================================
# Adult configuration
# ============================================================

config = RunConfig(
    dataset_name="adult",
    model_name="tabsyn",
    categorical_cols=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    continuous_cols=[
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    target_col_raw="class",
    constraints={},
)
# ============================================================
# Preprocess and split
# ============================================================

train_df, test_df, target_col, paths = preprocess_and_split(config)


# TabSyn utils.py requires NumPy files.
prepare_tabsyn_data(
    train_df,
    test_df,
    target_col,
    paths,
    config.continuous_cols,
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
    config.continuous_cols,
    config.categorical_cols,
    target_col,
)

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)
# ============================================================
# Evaluate
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
# Runtime and system summary
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
