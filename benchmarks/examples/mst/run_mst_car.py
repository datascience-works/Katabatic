import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.mst.models import MSTModel

config = RunConfig(
    dataset_name="car",
    model_name="mst",
    categorical_cols=["0", "1", "2", "3", "4", "5"],
    continuous_cols=[],
    target_col_raw="6",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train MST")
print("=" * 60)
model = MSTModel(categorical_columns=config.categorical_cols + [target_col])
model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
)
print("\nMST training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
