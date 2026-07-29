from __future__ import annotations

import os
import pandas as pd
import numpy as np

from typing import Optional, Dict, Any

from katabatic.models.base_model import Model
from sdv.single_table import CopulaGANSynthesizer

from .utils import build_single_table_metadata


class CopulaGANModel(Model):
    """
    Katabatic-compatible CopulaGAN Model
    """

    DATASET_LABEL_MAP = {
        "adult": "income",
        "magic": "class",
        "nursery": "8",
        "shuttle": "class",
        "car": "6",
    }

    def __init__(
        self,
        target_col: str = "target",
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        verbose: bool = False,
        pac: int = 10,
        enable_gpu: bool = True,
        numerical_distributions: Optional[Dict[str, str]] = None,
        default_distribution: str = "beta",
    ):
        super().__init__()

        self.target_col = target_col
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.pac = pac
        self.enable_gpu = enable_gpu
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution

        self._synth = None
        self._metadata = None

    def train(self, dataset_dir: str, synthetic_dir: str = None, **kwargs):

        dataset_name = dataset_dir.rstrip("/").split("/")[-1]

        if dataset_name not in self.DATASET_LABEL_MAP:
            raise ValueError(f"Unknown dataset {dataset_name}")

        label_col = self.DATASET_LABEL_MAP[dataset_name]

        x_train = pd.read_csv(os.path.join(dataset_dir, "x_train.csv"))
        y_df = pd.read_csv(os.path.join(dataset_dir, "y_train.csv"))

        if str(label_col) not in y_df.columns:
            y_df.columns = [str(c) for c in y_df.columns]

        y_train = y_df[str(label_col)]

        train_df = x_train.copy()
        train_df[self.target_col] = y_train.values

        self._metadata = build_single_table_metadata(
            train_df=train_df,
            target_col=self.target_col
        )

        self._synth = CopulaGANSynthesizer(
            metadata=self._metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            pac=self.pac,
            enable_gpu=self.enable_gpu,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution,
        )

        self._synth.fit(train_df)

        synth_df = self._synth.sample(num_rows=len(train_df))

        y_synth = synth_df[self.target_col]
        x_synth = synth_df.drop(columns=[self.target_col])

        os.makedirs(synthetic_dir, exist_ok=True)

        x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
        y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        self.is_fitted = True
        return self

    def sample(self, n_samples: int):
        if self._synth is None:
            raise RuntimeError("Model not trained")

        synth_df = self._synth.sample(num_rows=n_samples)

        y_synth = synth_df[self.target_col]
        x_synth = synth_df.drop(columns=[self.target_col])

        return x_synth.to_numpy(), y_synth.to_numpy()

    def evaluate(self, *args, **kwargs):
        return None
