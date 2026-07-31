import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from katabatic.models.ctabganplus.models import CTABGANPlus
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

config = RunConfig(
    dataset_name="car",
    model_name="ctabganplus",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train CTAB-GAN+")
print("=" * 60)
model = CTABGANPlus(
    config={
        # --- Training ---
        "epochs": 150,  # number of training epochs (default: 300)
        "batch_size": 500,  # samples per training batch (default: 500)
        # --- Architecture ---
        "random_dim": 100,  # noise vector size fed to the generator (default: 100)
        "num_channels": 64,  # base channel count for CNN layers; generator uses 4x, discriminator uses 1x and 2x (default: 64)
        "class_dim": (
            256,
            256,
            256,
            256,
        ),  # hidden layer sizes of the auxiliary classifier (default: (256, 256, 256, 256))
        # --- Regularisation ---
        "l2scale": 1e-5,  # L2 weight decay applied to all optimisers (default: 1e-5)
        # --- Column type overrides (leave empty if not needed) ---
        "log_columns": [],  # columns to apply log-transform before training (skewed distributions)
        "mixed_columns": {},  # columns with special discrete modal values, e.g. {"col": [0, -1]}
        "general_columns": [],  # continuous columns to scale linearly instead of fitting a GMM
        "non_categorical_columns": [],  # columns listed in categorical_cols that should stay continuous
        "integer_columns": [],  # columns to round to integers in the synthetic output
    }
)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nCTAB-GAN+ training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
