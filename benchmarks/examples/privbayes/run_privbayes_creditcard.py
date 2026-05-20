import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.privbayes.models import PrivBayesModel

config = RunConfig(
    dataset_name     = "creditcard_privbayes",
    model_name       = "privbayes",
    categorical_cols = [],
    continuous_cols  = ["V3", "V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "Amount"],
    target_col_raw   = "Class",
    constraints      = {"Amount": (0.0, 25691.16)},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)
print("STEP 3 — Train PrivBayes")
model = PrivBayesModel(epsilon=1.0, degree=2, seed=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("PrivBayes training complete.")
print("STEP 4 — Generate synthetic data")
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
