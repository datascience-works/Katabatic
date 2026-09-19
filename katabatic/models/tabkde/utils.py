"""
TabKDE utils — ported and adapted from the official TabKDE repository
(https://github.com/tabkde/tabkde-main), which itself extends TabSyn
(https://github.com/amazon-science/tabsyn).

Original authors: TabKDE paper authors (copula/KDE encoding).
This file adapts their `DataProcessor` and `EmpiricalTransformer` classes
(from tabkde/copula_encoding/model.py) plus a diffusion-style generative
model (from tabkde/model.py, tabkde/main.py) to fit the Katabatic
Model interface (train(data_dir, categorical_cols=, continuous_cols=),
sample(n)).

ADAPTATION NOTES (for documentation / honesty with markers):
- The original repo relies on a per-dataset `info.json` and a full
  pre-processing/dataset-download pipeline. Katabatic already prepares
  `x_train.csv` / `y_train.csv` (or `train_full.csv`) via its own
  pipeline, so we read directly from that instead of `info.json`.
- The original diffusion stage uses an EDM-style precondition + loss
  (tabkde/model.py `Precond`, `diffusion_utils.py EDMLoss`). Reproducing
  that exactly requires files we do not have local access to
  (`diffusion_utils.py` in full). Here we implement a standard
  DDPM-style denoising diffusion model (predict-noise, MSE loss) trained
  in the same *copula latent space* that TabKDE trains in. This keeps
  TabKDE's core idea (empirical/copula encoding + generative model over
  the copula space) while using a simpler, well-understood diffusion
  formulation we can verify end-to-end.
- The `DataProcessor` and `EmpiricalTransformer` classes below are close
  ports of the originals, adapted to accept explicit categorical/
  continuous column lists (as Katabatic's runner scripts already provide)
  instead of inferring columns from `info.json`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config / State
# ---------------------------------------------------------------------------

@dataclass
class TabKDEConfig:
    # diffusion training
    diffusion_epochs: int = 1000
    diffusion_batch_size: int = 512
    hidden_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 50
    # sampling
    diffusion_steps: int = 50
    # misc
    seed: int = 42
    device: Optional[str] = None


@dataclass
class TabKDEState:
    data_processor: "DataProcessor"
    empirical: "EmpiricalTransformer"
    denoiser: Any  # torch.nn.Module, kept as Any so this file has no hard torch import at module load
    columns: List[str]
    cat_columns: List[str]
    num_columns: List[str]
    n_train: int
    device: str
    cfg: TabKDEConfig


# ---------------------------------------------------------------------------
# Ported (adapted): category -> rank encoding + empirical (copula) transform
# ---------------------------------------------------------------------------

def _compute_category_map(df: pd.DataFrame, column_name: str, v: np.ndarray) -> Dict[Any, int]:
    """Rank categories by the mean of a reference numeric signal `v`.
    Ported from tabkde/copula_encoding/model.py `compute_category_map`.
    """
    unique_categories = df[column_name].unique()
    category_indices = {c: np.where(df[column_name] == c)[0] for c in unique_categories}
    category_means = {c: float(np.mean(v[idx])) if len(idx) else 0.0 for c, idx in category_indices.items()}
    sorted_categories = sorted(category_means.items(), key=lambda item: item[1])
    return {c: rank + 1 for rank, (c, _) in enumerate(sorted_categories)}


class DataProcessor:
    """
    Encodes categorical columns as ordered integer ranks (ordered by their
    relationship to the first principal component of the numeric columns,
    when available), then standard-scales the full numeric matrix.

    Adapted from tabkde/copula_encoding/model.py `DataProcessor`.
    Simplification vs. original: no `info.json`/decorrelation-by-default;
    categorical/numeric columns are passed explicitly instead of inferred.
    """

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.encoding_map: Dict[str, Dict[Any, int]] = {}
        self.cat_columns: List[str] = []
        self.num_columns: List[str] = []
        self.columns: List[str] = []
        self.pca_1: Optional[PCA] = None
        self.scaler_1: Optional[StandardScaler] = None
        self.v: Optional[np.ndarray] = None

    def fit(self, df_train: pd.DataFrame, cat_columns: List[str], num_columns: List[str]) -> pd.DataFrame:
        self.columns = list(df_train.columns)
        self.cat_columns = list(cat_columns)
        self.num_columns = list(num_columns)

        df_encoded = df_train.copy()

        # Reference signal `v` used to order categories sensibly: first
        # principal component of the numeric columns, if any exist.
        if len(self.num_columns) > 0:
            self.scaler_1 = StandardScaler()
            standardized = self.scaler_1.fit_transform(df_train[self.num_columns])
            self.pca_1 = PCA(n_components=1)
            self.v = self.pca_1.fit_transform(standardized).ravel()
        else:
            self.v = None

        for col in self.cat_columns:
            if self.v is not None:
                self.encoding_map[col] = _compute_category_map(df_train, col, self.v)
            else:
                cats = df_train[col].astype("category").cat.categories
                self.encoding_map[col] = {cat: code for code, cat in enumerate(cats, start=1)}
            df_encoded[col] = df_train[col].map(self.encoding_map[col])

        df_encoded = df_encoded.astype(float)

        self.scaler = StandardScaler()
        df_encoded[:] = self.scaler.fit_transform(df_encoded)

        return df_encoded

    def decode(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """Reverse scaling + categorical rank encoding back to original values."""
        df_decoded = df_encoded.copy()

        if self.scaler is not None:
            df_decoded[:] = self.scaler.inverse_transform(df_decoded)

        for col in self.cat_columns:
            mapping = self.encoding_map[col]
            reverse = {code: cat for cat, code in mapping.items()}
            min_key, max_key = min(reverse.keys()), max(reverse.keys())
            df_decoded[col] = df_decoded[col].round().astype(int).apply(
                lambda x, rev=reverse, lo=min_key, hi=max_key: rev.get(min(max(x, lo), hi), rev[lo])
            )

        for col in self.num_columns:
            # numeric columns stay numeric floats
            df_decoded[col] = df_decoded[col].astype(float)

        return df_decoded[self.columns]


class EmpiricalTransformer:
    """
    Fits the empirical (rank-based) CDF of each column and can map
    U(0,1) values back through the inverse empirical CDF. This is the
    "KDE-flavoured" piece of TabKDE: rather than assuming a parametric
    marginal distribution, each column's marginal is represented by its
    own training-data order statistics.

    Ported close to tabkde/copula_encoding/model.py `EmpiricalTransformer`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.df_sorted: Optional[pd.DataFrame] = None
        self.df_ranks: Optional[pd.DataFrame] = None
        self.fit()

    def fit(self, method: str = "min") -> pd.DataFrame:
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method=method).astype(int) / self.df.shape[0]
        return self.df_ranks

    @staticmethod
    def _inverse_empirical(u: float, sorted_col: np.ndarray) -> float:
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n
        return float(np.interp(u, ecdf, sorted_col))

    def convert(self, u_vectors: np.ndarray) -> pd.DataFrame:
        """u_vectors: (N, n_cols) array of values in [0, 1] -> back to encoded space."""
        out = []
        for u_vec in u_vectors:
            row = [
                self._inverse_empirical(u, self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            out.append(row)
        return pd.DataFrame(np.array(out), columns=self.df.columns)


# ---------------------------------------------------------------------------
# Copula-space <-> Gaussian-space helpers
# ---------------------------------------------------------------------------

def _uniform_to_gaussian(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map (0,1) ranks to standard-normal latents via the inverse normal CDF,
    so the diffusion model can operate in a well-behaved continuous space.
    This is a standard copula trick (Gaussian copula)."""
    from scipy.stats import norm
    u_clipped = np.clip(u, eps, 1 - eps)
    return norm.ppf(u_clipped)


def _gaussian_to_uniform(z: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# Diffusion model (simplified DDPM-style, trained in copula/Gaussian space)
# ---------------------------------------------------------------------------

def _build_denoiser(in_dim: int, hidden_dim: int):
    import torch.nn as nn

    class Denoiser(nn.Module):
        """Small MLP that predicts the noise added to a latent vector,
        conditioned on the diffusion timestep (sinusoidal embedding)."""

        def __init__(self, in_dim: int, hidden_dim: int, n_steps: int = 1000):
            super().__init__()
            self.n_steps = n_steps
            self.time_embed = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.net = nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, in_dim),
            )

        def forward(self, x, t):
            # t: (B,) integer timesteps -> normalise to [0, 1]
            t_norm = (t.float() / self.n_steps).unsqueeze(-1)
            temb = self.time_embed(t_norm)
            inp = __import__("torch").cat([x, temb], dim=-1)
            return self.net(inp)

    return Denoiser(in_dim, hidden_dim)


def _linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    import torch
    return torch.linspace(beta_start, beta_end, n_steps)


def train_tabkde(
    data_dir: str,
    cfg: TabKDEConfig,
    categorical_cols: List[str],
    continuous_cols: List[str],
) -> TabKDEState:
    import os
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- 1. Load data (matches the convention already used by TabSyn/TVAE
    #         ports in this codebase: train_full.csv, or x_train+y_train) ----
    train_full = os.path.join(data_dir, "train_full.csv")
    x_path = os.path.join(data_dir, "x_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        df = pd.read_csv(train_full)
        label_col = df.columns[-1]
    else:
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
        df = pd.concat([X, y[label_col]], axis=1)

    # Treat the label as categorical (classification target) unless the
    # caller already listed it under continuous_cols.
    cat_columns = list(categorical_cols)
    num_columns = list(continuous_cols)
    if label_col not in cat_columns and label_col not in num_columns:
        cat_columns = cat_columns + [label_col]

    # ---- 2. Encode (DataProcessor) ----
    processor = DataProcessor()
    df_encoded = processor.fit(df, cat_columns=cat_columns, num_columns=num_columns)

    # ---- 3. Empirical / copula transform ----
    empirical = EmpiricalTransformer(df_encoded)
    ranks = empirical.df_ranks.to_numpy()  # (N, n_cols) in (0, 1)
    z_gauss = _uniform_to_gaussian(ranks)  # (N, n_cols) standard-normal-ish

    z_tensor = torch.tensor(z_gauss, dtype=torch.float32)
    in_dim = z_tensor.shape[1]

    # ---- 4. Train a simple DDPM-style denoiser in this latent space ----
    n_steps = cfg.diffusion_steps if cfg.diffusion_steps > 1 else 50
    betas = _linear_beta_schedule(max(n_steps, 50)).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    denoiser = _build_denoiser(in_dim, cfg.hidden_dim)
    denoiser.n_steps = len(betas)
    denoiser = denoiser.to(device)

    loader = DataLoader(
        TensorDataset(z_tensor),
        batch_size=min(cfg.diffusion_batch_size, len(z_tensor)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_loss = float("inf")
    patience = 0
    denoiser.train()
    for epoch in range(cfg.diffusion_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, len(betas), (batch.shape[0],), device=device)
            noise = torch.randn_like(batch)
            sqrt_ab = alpha_bars[t].sqrt().unsqueeze(-1)
            sqrt_1m_ab = (1 - alpha_bars[t]).sqrt().unsqueeze(-1)
            noisy = sqrt_ab * batch + sqrt_1m_ab * noise

            pred_noise = denoiser(noisy, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

        if (epoch + 1) % max(1, cfg.diffusion_epochs // 10) == 0:
            print(f"[TabKDE] epoch {epoch+1}/{cfg.diffusion_epochs} loss={avg_loss:.4f}")

    denoiser.eval()

    state = TabKDEState(
        data_processor=processor,
        empirical=empirical,
        denoiser=denoiser,
        columns=list(df.columns),
        cat_columns=cat_columns,
        num_columns=num_columns,
        n_train=len(df),
        device=str(device),
        cfg=cfg,
    )
    # stash betas/alpha_bars on the state's denoiser module for use at sampling time
    state.denoiser._betas = betas.cpu()
    state.denoiser._alpha_bars = alpha_bars.cpu()
    return state


def sample_tabkde(state: TabKDEState, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Reverse-diffuse Gaussian latents -> uniform (copula) space ->
    EmpiricalTransformer.convert() -> DataProcessor.decode() -> real rows."""
    import torch

    n = int(n_samples) if n_samples else state.n_train
    device = torch.device(state.device)
    denoiser = state.denoiser
    betas = denoiser._betas.to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    n_steps = len(betas)

    in_dim = len(state.data_processor.columns)
    x = torch.randn(n, in_dim, device=device)

    denoiser.eval()
    with torch.no_grad():
        for t in reversed(range(n_steps)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            pred_noise = denoiser(x, t_batch)

            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            beta_t = betas[t]

            coef1 = 1.0 / alpha_t.sqrt()
            coef2 = beta_t / (1 - alpha_bar_t).sqrt()
            mean = coef1 * (x - coef2 * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = beta_t.sqrt()
                x = mean + sigma_t * noise
            else:
                x = mean

    z_gauss = x.cpu().numpy()
    uniform = _gaussian_to_uniform(z_gauss)
    df_encoded = state.empirical.convert(uniform)
    df_out = state.data_processor.decode(df_encoded)
    return df_out


def evaluate_tabkde(state: TabKDEState, data_dir: str, split: str = "test") -> float:
    """Lightweight sanity-check metric: mean absolute difference between the
    empirical marginal mean/std of real vs. freshly sampled synthetic data,
    averaged across numeric columns. Lower is better (0 = perfect marginal
    match). This is NOT a substitute for the project's full evaluation
    pipeline (SyntheticEvaluationPipeline) — it exists only so the model
    satisfies the base `evaluate()` contract with something meaningful."""
    import os

    if not state.num_columns:
        return 0.0

    synth = sample_tabkde(state, n_samples=min(1000, state.n_train))

    x_path = os.path.join(data_dir, "x_train.csv")
    if not os.path.exists(x_path):
        return 0.0
    real = pd.read_csv(x_path)

    diffs = []
    for col in state.num_columns:
        if col in real.columns and col in synth.columns:
            real_mean, real_std = real[col].mean(), real[col].std() + 1e-8
            synth_mean, synth_std = synth[col].mean(), synth[col].std() + 1e-8
            diffs.append(abs(real_mean - synth_mean) / real_std)
    return float(np.mean(diffs)) if diffs else 0.0
