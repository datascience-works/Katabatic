import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
from katabatic.models.tabpfgen.models import TabPFGenModel
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

# max_train_rows=1000: TabPFN is designed for small datasets; this also keeps
# generation fast. runner.py caps x_train.csv to this size.
config = RunConfig(
    dataset_name="adult",
    model_name="tabpfgen",
    categorical_cols=[
        "workclass",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="income",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
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
