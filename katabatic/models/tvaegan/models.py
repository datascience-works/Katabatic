"""Katabatic wrapper for a self-contained TVAE-GAN model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from katabatic.models.base_model import Model
from .utils import TVAEGANSynthesizer

logger = logging.getLogger(__name__)


class TVAEGANModel(Model):
    """Katabatic wrapper for the T-VAE-GAN tabular synthesizer.

    Expected Katabatic train split layout:
        dataset_dir/
            x_train.csv
            y_train.csv

    Training writes:
        synthetic_dir/
            x_synth.csv
            y_synth.csv
            synthetic.csv
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 500,
        cat_emb_size: int = 25,
        num_emb_size: int = 25,
        w_regularize: float = 1.0,
        w_reconstruct: float = 10.0,
        s_generat: int = 5,
        s_encoder: int = 5,
        lr_generat: float = 5e-5,
        lr_critic: float = 5e-5,
        lr_encoder: float = 5e-5,
        clip: float = 0.01,
        dropout: float = 0.1,
        hidden_layers_multipliers: Optional[list[float]] = None,
        shuffle: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.synth_params = dict(
            epochs=epochs,
            batch_size=batch_size,
            cat_emb_size=cat_emb_size,
            num_emb_size=num_emb_size,
            w_regularize=w_regularize,
            w_reconstruct=w_reconstruct,
            s_generat=s_generat,
            s_encoder=s_encoder,
            lr_generat=lr_generat,
            lr_critic=lr_critic,
            lr_encoder=lr_encoder,
            clip=clip,
            dropout=dropout,
            hidden_layers_multipliers=hidden_layers_multipliers or [1.0, 1.0],
            shuffle=shuffle,
            random_state=random_state,
        )

        self.synthesizer: Optional[TVAEGANSynthesizer] = None
        self.y_col_name_: Optional[str] = None
        self.valid_classes_: Optional[list] = None
        self.y_dtype_ = None
        self.is_fitted = False

    def train(self, dataset_dir: str | Path, synthetic_dir: str | Path, **kwargs) -> "TVAEGANModel":
        """Train TVAE-GAN and write Katabatic-compatible synthetic outputs."""
        dataset_dir = Path(dataset_dir)
        synthetic_dir = Path(synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        x_train_path = dataset_dir / "x_train.csv"
        y_train_path = dataset_dir / "y_train.csv"

        if not x_train_path.exists():
            raise FileNotFoundError(f"x_train.csv not found at {x_train_path}")
        if not y_train_path.exists():
            raise FileNotFoundError(f"y_train.csv not found at {y_train_path}")

        X_train = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)

        self.y_col_name_ = y_train_df.columns[0]
        y_train = y_train_df[self.y_col_name_]

        df_train = X_train.copy()
        df_train[self.y_col_name_] = y_train.values

        logger.info("Training TVAEGANModel on shape %s", df_train.shape)
        self.synthesizer = TVAEGANSynthesizer(
            **self.synth_params,
            forced_cat_columns=[self.y_col_name_]
        )
        self.synthesizer.fit(df_train)

        df_synth = self.synthesizer.predict(len(df_train))

        self.valid_classes_ = sorted(pd.Series(y_train).dropna().unique())
        self.y_dtype_ = y_train.dtype

        df_synth[self.y_col_name_] = (
            df_synth[self.y_col_name_]
            .apply(self._snap_label)
            .astype(self.y_dtype_)
        )

        x_synth = df_synth.drop(columns=[self.y_col_name_])
        y_synth = df_synth[[self.y_col_name_]]

        x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_synth.to_csv(synthetic_dir / "y_synth.csv", index=False)
        df_synth.to_csv(synthetic_dir / "synthetic.csv", index=False)

        self.is_fitted = True
        return self

    def _snap_label(self, value):
        if value in self.valid_classes_:
            return value
        try:
            rounded = int(round(float(value)))
            return min(self.valid_classes_, key=lambda c: abs(c - rounded))
        except (TypeError, ValueError):
            return value

    def sample(self, n: int, seed: int = None, **kwargs) -> pd.DataFrame:
        if self.synthesizer is None or not self.is_fitted:
            raise RuntimeError("TVAEGANModel must be trained before calling sample().")
        df = self.synthesizer.predict(n, seed=seed)
        if self.valid_classes_ is not None and self.y_col_name_ in df.columns:
            df[self.y_col_name_] = (
                df[self.y_col_name_]
                .apply(self._snap_label)
                .astype(self.y_dtype_)
            )
        return df

