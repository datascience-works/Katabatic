import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.experimental.models.medgan.models import MEDGAN

config = RunConfig(
    dataset_name="shuttle",
    model_name="medgan",
    categorical_cols=[],
    continuous_cols=["time", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    target_col_raw="class",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train MedGAN")
print("=" * 60)
model = MEDGAN(ae_pretrain_epochs=100, gan_epochs=1000, batch_size=256, random_state=42)
model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nMedGAN training complete.")

target_values = np.sort(train_df[target_col].unique())


def sample_dataframe(n, seed=None):
    """Return the raw MedGAN samples with valid Shuttle class labels."""
    if seed is not None:
        torch.manual_seed(seed)
    frame = pd.DataFrame(model.sample(n), columns=train_df.columns)
    nearest = np.abs(frame[target_col].to_numpy()[:, None] - target_values).argmin(
        axis=1
    )
    frame[target_col] = target_values[nearest]
    return frame


print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = save_synthetic(
    sample_dataframe(len(train_df)),
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)

# Stability also needs samples as DataFrames in the original dataset schema.
evaluate(
    SimpleNamespace(sample=sample_dataframe),
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)
