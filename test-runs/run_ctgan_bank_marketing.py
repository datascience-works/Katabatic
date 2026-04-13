import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.ctgan.models import CTGANModel

config = RunConfig(
    dataset_name     = "bank_marketing",
    model_name       = "ctgan",
    categorical_cols = ['1', '2', '3', '4', '6', '7', '8', '10', '15'],  # job, marital, education, default, housing, loan, contact, month, poutcome
    continuous_cols  = ['0', '5', '9', '11', '12', '13', '14'],           # age, balance, day, duration, campaign, pdays, previous
    target_col_raw   = "y",
    constraints      = {
        '0':  (18, 95),      # age: legal working/banking age range
        '5':  (-8020, None), # balance: min observed in dataset, no upper bound
        '9':  (1, 31),       # day: day of month
        '11': (0, None),     # duration: call duration in seconds, cannot be negative
        '12': (1, None),     # campaign: at least 1 contact was made
        '13': (-1, None),    # pdays: -1 = not previously contacted, otherwise >= 0
        '14': (0, None),     # previous: number of previous contacts, cannot be negative
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train CTGAN")
print("=" * 60)
model = CTGANModel(epochs=100, batch_size=512, seed=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols,
                                continuous_cols=config.continuous_cols)
print("\nCTGAN training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths)

evaluate(model, config, train_df, synthetic_df, target_col, paths)
