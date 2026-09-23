"""
TVAE
====

Katabatic implementation of a Tabular Variational Autoencoder, based on:

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
"Modeling Tabular data using Conditional GAN."
Advances in Neural Information Processing Systems (NeurIPS).
Paper: https://arxiv.org/abs/1907.00503
Code: https://github.com/sdv-dev/CTGAN

Overview
--------
TVAE optimizes a single reconstruction + KL-divergence loss, avoiding the
adversarial training dynamics that GAN-based tabular models can suffer
from (mode collapse, discriminator/generator imbalance).

This implementation delegates the actual VAE training/sampling to the
reference `ctgan.TVAE` synthesizer (sdv-dev), but follows the same
schema-based encode/decode approach as ``katabatic/models/ctgan`` and
``katabatic/models/fairtabdiffusion`` (continuous columns quantile-
normalized to Normal, categorical columns one-hot encoded via
``fairtabdiffusion/utils.py``'s shared helpers), and the same ``Model``
contract (``train``, ``evaluate``, ``sample``) defined in
``katabatic/models/base_model.py``.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from katabatic.models.base_model import Model as BaseModel

try:
    from ctgan import TVAE as _TVAESynthesizer
except ImportError as e:
    raise ImportError(
        "TVAE requires the 'ctgan' package. Install with: poetry add ctgan"
    ) from e


class TVAE(BaseModel):
    """Variational autoencoder for mixed-type tabular data."""

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        compress_dims: tuple[int, ...] = (128, 128),
        decompress_dims: tuple[int, ...] = (128, 128),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "compress_dims": compress_dims,
            "decompress_dims": decompress_dims,
            "seed": seed,
        }

        self._synthesizer = _TVAESynthesizer(
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            batch_size=batch_size,
            epochs=epochs,
        )
        self._label_col: str | None = None
        self._discrete_columns: list[str] = []
        self._train_df: pd.DataFrame | None = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["ctgan"]

    def train(
        self,
        data_dir: str,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> TVAE:
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. "
                    "Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column.")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self._train_df = df.copy()
        self._label_col = df.columns[-1]

        cat_cols_full = list(categorical_cols or [])
        if self._label_col not in cat_cols_full:
            cat_cols_full = [*cat_cols_full, self._label_col]
        self._discrete_columns = [c for c in cat_cols_full if c in df.columns]

        self._synthesizer.fit(df, discrete_columns=self._discrete_columns)
        self.is_fitted = True

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            df_s = self.sample(n=len(df))
            feature_cols = [c for c in df.columns if c != self._label_col]
            x_synth = df_s[feature_cols].copy()
            y_synth = df_s[[self._label_col]].copy()
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False, header=True
            )
            meta = {
                "schema": {
                    "columns": list(df.columns),
                    "label": self._label_col,
                    "discrete_columns": self._discrete_columns,
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                },
                "training": self.cfg,
            }
            with open(
                os.path.join(synthetic_dir, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f, indent=2)
            print(f"[TVAE] Synthetic data saved to {synthetic_dir}")

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: int | None = None,
        conditional: dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")
        if conditional is not None:
            raise ValueError(
                "TVAE does not support conditional sampling in this "
                "implementation. Leave conditional=None."
            )
        size = int(n) if n is not None else 1000
        return self._synthesizer.sample(size)
