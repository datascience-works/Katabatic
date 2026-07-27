import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.pacgan.models import CWGANGP

config = RunConfig(
    dataset_name     = "adult",
    model_name       = "cwgangp",
    categorical_cols = ['workclass', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'],
    continuous_cols  = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
    target_col_raw   = "income",
    constraints      = {
        'age':          (17, 90),          # working age range
        'fnlwgt':       (12285, 1490400),  # census sampling weight, dataset min/max
        'capital-gain': (0, 99999),        # cannot be negative, capped at 99999 in dataset
        'capital-loss': (0, 4356),         # cannot be negative, capped at 4356 in dataset
        'hours-per-week': (1, 99),         # at least 1 hour, max 99 in dataset
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train CWGAN-GP")
print("=" * 60)
model = CWGANGP(niter=500, batch_size=256, z_dim=64, random_state=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("\nCWGAN-GP training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
