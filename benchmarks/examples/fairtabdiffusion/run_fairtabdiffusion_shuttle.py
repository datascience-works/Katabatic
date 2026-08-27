import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

# Statlog (Shuttle) dataset column layout (UCI):
# ['time', 'Rad Flow', 'Fpv Close', 'Fpv Open', 'High', 'Bypass',
#  'Bpv Close', 'Bpv Open', 'class']
# All 8 features are continuous/integer-valued; 'class' is the target
# with 7 levels (heavily imbalanced -- ~80% of rows are class 1).
# NOTE: verify column names against your team's raw_data/shuttle.csv --
# some distributions ship without a header row or with different column
# naming (e.g. "col1".."col9"). Check with:
#   head -1 raw_data/shuttle.csv
# If there's no header, you'll need to add one before this script's
# preprocess_and_split call will find 'class' as target_col_raw.
config = RunConfig(
    dataset_name="shuttle",
    model_name="fairtabdiffusion",
    categorical_cols=[],
    continuous_cols=[
        "time",
        "Rad Flow",
        "Fpv Close",
        "Fpv Open",
        "High",
        "Bypass",
        "Bpv Close",
        "Bpv Open",
    ],
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
