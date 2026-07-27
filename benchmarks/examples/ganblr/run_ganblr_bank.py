import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.ganblr.models import GANBLR

config = RunConfig(
    dataset_name     = "bank_marketing",
    model_name       = "ganblr",
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'],
    continuous_cols  = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
    target_col_raw   = "y",
    constraints      = {
        'age':      (18, 95),      # legal working/banking age range
        'balance':  (-8020, None), # min observed in dataset, no upper bound
        'day':      (1, 31),       # day of month
        'duration': (0, None),     # call duration in seconds, cannot be negative
        'campaign': (1, None),     # at least 1 contact was made
        'pdays':    (-1, None),    # -1 = not previously contacted, otherwise >= 0
        'previous': (0, None),     # number of previous contacts, cannot be negative
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train GANBLR")
print("=" * 60)
model = GANBLR()
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
    epochs=150,
    batch_size=32,
    k=0,
    warmup_epochs=1,
)
print("\nGANBLR training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60) 
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
