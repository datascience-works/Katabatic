import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

config = RunConfig(
    dataset_name="car",
    model_name="fairtabdiffusion",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FairTabDiffusion")
print("=" * 60)
# NOTE: Car has no natural sensitive attribute, so sensitive_col is left
# unset -- the model falls back to unconditional (label-only) diffusion.
# For the Adult dataset, set sensitive_col="sex" (or "race") instead.
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
