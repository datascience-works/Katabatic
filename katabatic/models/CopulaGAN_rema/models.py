

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

# SDV CopulaGAN
from sdv.single_table import CopulaGANSynthesizer

from .utils.metadata import build_single_table_metadata


@dataclass
class CopulaGANModel:
    """
    Katabatic wrapper around SDV's CopulaGANSynthesizer.

    Strategy (simple + stable):
    - Combine X + y into one dataframe
    - Create SDV Metadata
    - Fit CopulaGAN
    - Sample full rows
    - Split back into x_synth and y_synth
    """

    target_col: str = "target"

    # main training knobs
    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    generator_dim: tuple = (256, 256)
    discriminator_dim: tuple = (256, 256)
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    discriminator_steps: int = 1
    log_frequency: bool = True
    verbose: bool = False
    pac: int = 10
    enable_gpu: bool = True

    # copula options
    numerical_distributions: Optional[Dict[str, str]] = None
    default_distribution: str = "beta"

    _synth: Optional[CopulaGANSynthesizer] = None
    _metadata: Optional[Any] = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        train_df = self._combine_xy(x_train, y_train)

        # Build SDV metadata (single table)
        self._metadata = build_single_table_metadata(
            train_df=train_df,
            target_col=self.target_col
        )

        # Create SDV CopulaGAN synthesizer
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
            numerical_distributions=self.numerical_distributions or {},
            default_distribution=self.default_distribution,
        )

        self._synth.fit(train_df)

    def sample(self, num_rows: int) -> tuple[pd.DataFrame, pd.Series]:
        if self._synth is None:
            raise RuntimeError("CopulaGANModel is not fitted. Call fit() first.")

        synth_df = self._synth.sample(num_rows=num_rows)

        if self.target_col not in synth_df.columns:
            raise RuntimeError(
                f"Sampled data missing target_col='{self.target_col}'. "
                "Check metadata building logic."
            )

        y_synth = synth_df[self.target_col]
        x_synth = synth_df.drop(columns=[self.target_col])

        return x_synth, y_synth

    def fit_and_sample(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        num_rows: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(x_train, y_train)
        n = int(num_rows) if num_rows is not None else len(x_train)
        return self.sample(n)

    def _combine_xy(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = x.copy()
        df[self.target_col] = y.reset_index(drop=True).values
        return df
