import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.arf.models import ARFModel

config = RunConfig(
    dataset_name="magic",
    model_name="arf",
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
print("STEP 3 - Train ARF")
print("=" * 60)
model = ARFModel(num_trees=10, max_iters=10)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nARF training complete.")

print("\n" + "=" * 60)
print("STEP 4 - Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
