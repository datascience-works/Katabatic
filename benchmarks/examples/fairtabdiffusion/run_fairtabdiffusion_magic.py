import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

# MAGIC Gamma Telescope dataset column layout (UCI):
# ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long',
#  'fM3Trans', 'fAlpha', 'fDist', 'class']
# All 10 features are continuous; 'class' is the binary target (g/h).
# NOTE: verify these column names against your team's raw_data/magic.csv —
# some versions of this dataset ship without a header row, in which case
# you'll need to add these names yourself before running this script
# (see the earlier Adult column-name mismatch for how that error looks).
config = RunConfig(
    dataset_name="magic",
    model_name="fairtabdiffusion",
    categorical_cols=[],
    continuous_cols=[
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
    ],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FairTabDiffusion")
print("=" * 60)
# MAGIC has no natural sensitive attribute (it's physics/telescope data),
# so sensitive_col is left unset, same as Car. This run's main purpose is
# to test FairTabDiffusion on an all-continuous feature space, isolating
# whether the consistency drop seen on Adult is tied to continuous
# features specifically or was something else about Adult's schema.
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
