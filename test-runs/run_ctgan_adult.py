import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.ctgan.models import CTGANModel

config = RunConfig(
    dataset_name     = "adult",
    model_name       = "ctgan",
    categorical_cols = ['1', '3', '4', '5', '6', '7', '8', '9', '13'],  # workclass, education, educational-num, marital-status, occupation, relationship, race, gender, native-country
    continuous_cols  = ['0', '2', '10', '11', '12'],                     # age, fnlwgt, capital-gain, capital-loss, hours-per-week
    target_col_raw   = "income",
    constraints      = {
        '0':  (17, 90),          # age: working age range
        '2':  (12285, 1490400),  # fnlwgt: census sampling weight, dataset min/max
        '10': (0, 99999),        # capital-gain: cannot be negative, capped at 99999 in dataset
        '11': (0, 4356),         # capital-loss: cannot be negative, capped at 4356 in dataset
        '12': (1, 99),           # hours-per-week: at least 1 hour, max 99 in dataset
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train CTGAN")
print("=" * 60)
model = CTGANModel(epochs=100, batch_size=512, seed=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("\nCTGAN training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths)

evaluate(model, config, train_df, synthetic_df, target_col, paths)
