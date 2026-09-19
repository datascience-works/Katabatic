"""
TVAE utils — wraps the official, published TVAE implementation from the
`ctgan` package (https://github.com/sdv-dev/CTGAN), maintained by the
original authors of the CTGAN/TVAE paper:

    Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
    "Modeling Tabular data using Conditional GAN." NeurIPS 2019.

TVAE itself is a direct application of the Variational Autoencoder (VAE)
framework from:

    Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes."

ADAPTATION NOTE: Unlike TabKDE (where no official maintained package
existed), TVAE has a real, actively-maintained, pip-installable official
implementation (pip install ctgan) published by the paper's own authors.
Rather than hand-reimplementing their Bayesian-GMM-based data transformer
and VAE encoder/decoder, this port wraps their official TVAE class directly
and adapts it to Katabatic's model interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TVAEConfig:
    embedding_dim: int = 128
    compress_dims: tuple = (128, 128)
    decompress_dims: tuple = (128, 128)
    l2scale: float = 1e-5
    batch_size: int = 500
    epochs: int = 300
    loss_factor: int = 2
    seed: int = 42
    enable_gpu: bool = False


@dataclass
class TVAEState:
    model: "object"
    columns: List[str]
    cat_columns: List[str]
    num_columns: List[str]
    n_train: int
    cfg: TVAEConfig


def _load_training_frame(data_dir: str) -> pd.DataFrame:
    """Matches the convention already used by the other model ports in this
    codebase (TabSyn, TabKDE): train_full.csv, or x_train.csv + y_train.csv."""
    train_full = os.path.join(data_dir, "train_full.csv")
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        return pd.read_csv(train_full)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Could not find training data in {data_dir}. "
            "Expected train_full.csv or x_train.csv/y_train.csv."
        )
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    if y.shape[1] != 1:
        raise ValueError("y_train.csv must have exactly one column (the target).")
    label_col = y.columns[0]
    return pd.concat([X, y[label_col]], axis=1)

def train_tvae(
    data_dir: str,
    cfg: TVAEConfig,
    categorical_cols: List[str],
    continuous_cols: List[str],
) -> TVAEState:
    from ctgan.synthesizers.tvae import TVAE as OfficialTVAE
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = _load_training_frame(data_dir)
    label_col = df.columns[-1]

    cat_columns = list(categorical_cols)
    num_columns = list(continuous_cols)
    if label_col not in cat_columns and label_col not in num_columns:
        cat_columns = cat_columns + [label_col]

    model = OfficialTVAE(
        embedding_dim=cfg.embedding_dim,
        compress_dims=cfg.compress_dims,
        decompress_dims=cfg.decompress_dims,
        l2scale=cfg.l2scale,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        loss_factor=cfg.loss_factor,
        enable_gpu=cfg.enable_gpu,
        verbose=True,
    )
    model.fit(df, discrete_columns=cat_columns)

    return TVAEState(
        model=model,
        columns=list(df.columns),
        cat_columns=cat_columns,
        num_columns=num_columns,
        n_train=len(df),
        cfg=cfg,
    )

def sample_tvae(state: TVAEState, n_samples: Optional[int] = None) -> pd.DataFrame:
    n = int(n_samples) if n_samples else state.n_train
    df_out = state.model.sample(n)
    return df_out[state.columns]


def evaluate_tvae(state: TVAEState, data_dir: str, split: str = "test") -> float:
    """Lightweight sanity-check metric (same spirit as the TabKDE port):
    mean absolute difference between real and freshly-sampled synthetic
    data's per-column mean/std for numeric columns, normalised by the real
    column's std. This is NOT a substitute for the project's full
    evaluation pipeline (SyntheticEvaluationPipeline)."""
    if not state.num_columns:
        return 0.0

    synth = sample_tvae(state, n_samples=min(1000, state.n_train))

    x_path = os.path.join(data_dir, "x_train.csv")
    if not os.path.exists(x_path):
        return 0.0
    real = pd.read_csv(x_path)

    diffs = []
    for col in state.num_columns:
        if col in real.columns and col in synth.columns:
            real_mean, real_std = real[col].mean(), real[col].std() + 1e-8
            synth_mean = synth[col].mean()
            diffs.append(abs(real_mean - synth_mean) / real_std)
    return float(np.mean(diffs)) if diffs else 0.0
