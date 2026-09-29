import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabpfgen.models import TabPFGenModel

# max_train_rows=1000: TabPFN is designed for small datasets; this also keeps
# generation fast. runner.py caps x_train.csv to this size.
config = RunConfig(
    dataset_name="bank_marketing",
    model_name="tabpfgen",
    categorical_cols=[
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ],
    continuous_cols=[
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ],
    target_col_raw="y",
    constraints={
        "age": (18, 95),  # legal working/banking age range
        "balance": (-8020, None),  # min observed in dataset, no upper bound
        "day": (1, 31),  # day of month
        "duration": (0, None),  # call duration in seconds, cannot be negative
        "campaign": (1, None),  # at least 1 contact was made
        "pdays": (-1, None),  # -1 = not previously contacted, otherwise >= 0
        "previous": (0, None),  # number of previous contacts, cannot be negative
    },
    max_train_rows=1000,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Generate synthetic data with TabPFGen")
print("=" * 60)
model = TabPFGenModel(
    task="classification",
    n_sgld_steps=300,
    sgld_step_size=0.01,
    sgld_noise_scale=0.01,
    balance_classes=True,
    device="auto",
    random_state=config.seed,
)
result = model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print(f"\nTabPFGen generation complete. SGLD steps: {result['n_sgld_steps']}")

print("\n" + "=" * 60)
print("STEP 4 — Load and validate synthetic data")
print("=" * 60)
synthetic_df = pd.read_csv(result["synthetic_csv"])
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
