import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

# katabatic/datasets/shuttle.csv has columns [a1..a8]
# All 9 features are continuous/integer-valued. 'class' is the target with
# 7 levels (heavily imbalanced with ~80% of rows being class 1).
config = RunConfig(
    dataset_name="shuttle",
    model_name="fairtabdiffusion",
    categorical_cols=[],
    continuous_cols=["time", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FairTabDiffusion")
print("=" * 60)
# Shuttle has no natural sensitive attribute (spacecraft telemetry data),
# so sensitive_col is left unset, same as Car, Magic, and Nursery.
# This is the second all-continuous test (after Magic) -- if the
# consistency drop reproduces here too, that further confirms the
# continuous-feature-handling hypothesis. Shuttle's heavy class
# imbalance (~80% class 1) is also worth watching in the utility results.
model = FairTabDiffusion(
    sensitive_col=None,
    epochs=200,
    timesteps=100,
    batch_size=256,
    seed=42,
)

model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

print("\nFairTabDiffusion training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
