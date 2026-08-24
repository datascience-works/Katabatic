import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.tabpfgen.models import TabPFGenModel

# max_train_rows=1000: TabPFN is designed for small datasets; this also keeps
# generation fast. runner.py caps x_train.csv to this size.
config = RunConfig(
    dataset_name     = "nursery",
    model_name       = "tabpfgen",
    categorical_cols=['parents','has_nurs','form','children','housing','finance','social','health'],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
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
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
