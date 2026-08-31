import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from katabatic.models.tabsyn.models import TabSyn
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from prepare_npy import prepare_npy_for_tabsyn

config = RunConfig(
    dataset_name="car",
    model_name="tabsyn",
    categorical_cols=["0", "1", "2", "3", "4", "5"],
    continuous_cols=[],
    target_col_raw="6",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

prepare_npy_for_tabsyn(paths["split_dir"], config.categorical_cols, config.continuous_cols)

print("\n" + "=" * 60)
print("STEP 3 : Train TabSyn")
print("=" * 60)
model = TabSyn(d_token=16, weight_decay=0.01, lr=5e-3, diffusion_steps=15)
model.train(paths["split_dir"], paths["synthetic_dir"])
print("\nTabSyn training complete.")

print("\n" + "=" * 60)
print("STEP 4 : Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
print("SYNTH COLUMNS (before rename):", synthetic_df.columns.tolist())

# Target was moved to FIRST categorical position by _concat_xy in utils.py,
# so rename must put target_col first, then reorder to match train_df.
synthetic_df.columns = [target_col] + list(train_df.columns[:-1])
synthetic_df = synthetic_df[train_df.columns.tolist()]
print("SYNTH COLUMNS (after rename):", synthetic_df.columns.tolist())

print(synthetic_df.head(10))
print("Synthetic label distribution:")
print(synthetic_df[target_col].value_counts())
print("Real label distribution:")
print(train_df[target_col].value_counts())

synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)