import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.tabebm.models import TabEBMModel

config = RunConfig(
    dataset_name     = "creditcard_privbayes",
    model_name       = "tabebm",
    categorical_cols = [],
    continuous_cols  = ["V3", "V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "Amount"],
    target_col_raw   = "Class",
    max_train_rows   = 10000,
    constraints      = {"Amount": (0.0, 25691.16)},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)
print("STEP 3 — Train TabEBM")
model = TabEBMModel(target_col=target_col)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("TabEBM training complete.")
print("STEP 4 — Generate synthetic data")
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
