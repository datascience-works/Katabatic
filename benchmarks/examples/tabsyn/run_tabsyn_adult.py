import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from katabatic.models.tabsyn.models import TabSyn
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from prepare_npy import prepare_npy_for_tabsyn

config = RunConfig(
    dataset_name="adult",
    model_name="tabsyn",
    categorical_cols=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

prepare_npy_for_tabsyn(paths["split_dir"], config.categorical_cols, config.continuous_cols)

print("\n" + "=" * 60)
print("STEP 3 : Train TabSyn")
print("=" * 60)
model = TabSyn()
model.train(paths["split_dir"], paths["synthetic_dir"])
print("\nTabSyn training complete.")

print("\n" + "=" * 60)
print("STEP 4 : Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
print("SYNTH COLUMNS (before rename):", synthetic_df.columns.tolist())

# Target was moved to FIRST categorical position by _concat_xy in utils.py.
# Numeric columns come first in the encoder's internal layout, then categoricals
# (with target as cat_0). Real train_df has features in original order + target last.
synthetic_df.columns = config.continuous_cols + [target_col] + config.categorical_cols
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