import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

# katabatic/datasets/nursery.csv uses positional headers ["0"-"8"]
# All 8 features + target are categorical.
config = RunConfig(
    dataset_name="nursery",
    model_name="fairtabdiffusion",
    categorical_cols=["0", "1", "2", "3", "4", "5", "6", "7"],
    continuous_cols=[],
    target_col_raw="8",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FairTabDiffusion")
print("=" * 60)
# Nursery has no natural sensitive attribute (it's school-admission
# criteria data), so sensitive_col is left unset, same as Car and Magic.
# This run tests FairTabDiffusion on a second all-categorical dataset,
# to check whether Car's strong consistency (0.80) generalizes to other
# all-categorical data, or was itself dataset-specific.
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
