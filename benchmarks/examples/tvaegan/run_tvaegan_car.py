import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from katabatic.models.tvaegan.models import TVAEGANModel
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

config = RunConfig(
    dataset_name="car",
    model_name="tvaegan",
    categorical_cols=["0", "1", "2", "3", "4", "5"],
    continuous_cols=[],
    target_col_raw="6",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 : Train TVAE-GAN")
print("=" * 60)
model = TVAEGANModel()
model.train(paths["split_dir"], paths["synthetic_dir"])
print("\nTVAE-GAN training complete.")

print("\n" + "=" * 60)
print("STEP 4 : Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
print("SYNTH COLUMNS:", synthetic_df.columns.tolist())
print(synthetic_df.head(10))

print("Synthetic label distribution:")
print(synthetic_df[target_col].value_counts())
print("Real label distribution:")
print(train_df[target_col].value_counts())

synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)