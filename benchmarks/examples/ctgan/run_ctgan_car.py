import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.ctgan.models import CTGANModel

config = RunConfig(
    dataset_name     = "car",
    model_name       = "ctgan",
    categorical_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols  = [],
    target_col_raw   = "class",
    constraints      = None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("STEP 3 - Train CTGAN")
model = CTGANModel(epochs=100, batch_size=512, seed=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("CTGAN training complete.")

print("STEP 4 - Generate synthetic data")
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
