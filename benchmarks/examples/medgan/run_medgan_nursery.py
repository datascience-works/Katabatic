import os
import sys
from tempfile import TemporaryDirectory
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
    dataset_name="nursery",
    model_name="medgan",
    categorical_cols=["0", "1", "2", "3", "4", "5", "6", "7"],
    continuous_cols=[],
    target_col_raw="8",
    constraints={},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# MedGAN casts its input to floats, so encode categories using training data
# only and decode samples before comparing them with the original data.
encoded_train = train_df.copy()
categories = {}
for column in config.categorical_cols + [target_col]:
    encoded_train[column], categories[column] = pd.factorize(
        train_df[column], sort=True
    )

print("\n" + "=" * 60)
print("STEP 3 — Train MedGAN")
print("=" * 60)
model = MEDGAN(ae_pretrain_epochs=100, gan_epochs=1000, batch_size=256, random_state=42)
with TemporaryDirectory(prefix="medgan-nursery-") as encoded_dir:
    encoded_train.drop(columns=[target_col]).to_csv(
        os.path.join(encoded_dir, "x_train.csv"), index=False
    )
    encoded_train[[target_col]].to_csv(
        os.path.join(encoded_dir, "y_train.csv"), index=False
    )
    model.train(
        encoded_dir,
        synthetic_dir=os.path.join(encoded_dir, "synthetic"),
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
print("\nMedGAN training complete.")


def sample_dataframe(n, seed=None):
    """Decode MedGAN's numeric array to the original Nursery categories."""
    if seed is not None:
        torch.manual_seed(seed)
    frame = pd.DataFrame(model.sample(n), columns=train_df.columns)
    for column, labels in categories.items():
        indices = np.rint(frame[column].to_numpy()).clip(0, len(labels) - 1).astype(int)
        frame[column] = labels.take(indices).to_numpy()
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

# Reuse decoding when the stability evaluator asks for additional samples.
evaluate(
    SimpleNamespace(sample=sample_dataframe),
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)
