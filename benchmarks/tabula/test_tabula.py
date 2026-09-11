import os

import pandas as pd
import torch

from benchmarks.runner import (
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)
from katabatic.models.registry import ModelRegistry

# ============================================================
# TEST CONFIGURATION
# ============================================================

DATASET = "adult"
MODEL = "tabula"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

config = RunConfig(
    dataset_name=DATASET,
    model_name=MODEL,
    categorical_cols=[],
    continuous_cols=[],
    target_col_raw="class",
    # Stage 2: bumped from 100 -> 1000, per staged smoke-test plan
    max_train_rows=1000,
    test_size=0.2,
    seed=42,
)


# ============================================================
# STEP 1 — PREPROCESS + SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TABULA TEST")
print("=" * 70)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\nTraining data:")
print(train_df.shape)

print("\nTest data:")
print(test_df.shape)

print("\nTarget column:")
print(target_col)


# ============================================================
# STEP 2 — CREATE TABULA MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — CREATE TABULA MODEL")
print("=" * 70)

model = ModelRegistry.create_model(MODEL)

print(f"Model created: {type(model).__name__}")


# ============================================================
# STEP 3 — TRAIN + GENERATE SYNTHETIC DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — TRAIN TABULA + GENERATE SYNTHETIC DATA")
print("=" * 70)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    # Stage 2 settings
    epochs=5,  # was 3
    n_samples=5,  # was 2
    k=1,
    max_length=128,  # unchanged, already correct
    max_rounds=20,  # was 5 — more attempts to find valid rows
)


# ============================================================
# STEP 4 — LOAD GENERATED DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — LOAD GENERATED DATA")
print("=" * 70)

x_synth_path = os.path.join(paths["synthetic_dir"], "x_synth.csv")
y_synth_path = os.path.join(paths["synthetic_dir"], "y_synth.csv")

if not os.path.exists(x_synth_path):
    raise FileNotFoundError(f"Expected synthetic feature data at:\n{x_synth_path}")

if not os.path.exists(y_synth_path):
    raise FileNotFoundError(f"Expected synthetic target data at:\n{y_synth_path}")

x_synth = pd.read_csv(x_synth_path)
y_synth = pd.read_csv(y_synth_path)

synthetic_df = pd.concat([x_synth, y_synth], axis=1)

synthetic_df = save_synthetic(synthetic_df, train_df, paths)

synthetic_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")

print("\nSynthetic data:")
print(synthetic_df.shape)

print("\nFirst 3 synthetic rows:")
print(synthetic_df.head(3).to_string())


# ============================================================
# STEP 5 — EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — EVALUATE TABULA")
print("=" * 70)

report = evaluate(
    model=model,
    config=config,
    train_df=train_df,
    synthetic_df=synthetic_df,
    target_col=target_col,
    paths=paths,
    test_df=test_df,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TABULA TEST COMPLETE")
print("=" * 70)

print(f"Dataset          : {DATASET}")
print(f"Model            : {MODEL}")
print(f"Synthetic data   : {synthetic_path}")
print(f"Results directory: {paths['results_dir']}")
print(f"Composite score  : {report.composite_score:.4f}")
