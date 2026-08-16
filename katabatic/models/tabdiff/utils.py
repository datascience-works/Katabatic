"""Joint-space encoding and a small denoising MLP for the TabDiff model.

TabDiff's core idea (vs. TabDDPM) is diffusing numeric and categorical columns
*jointly* in a single unified process, rather than running a separate Gaussian
process for numerics and a separate multinomial process for categoricals. This
module implements that joint process in the simplest faithful way: categorical
columns are one-hot encoded, numeric columns are z-scored, everything is
concatenated into one continuous vector, and a single Gaussian diffusion runs
over the whole thing. Categorical blocks are recovered with argmax at decode
time.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_column_roles(dataset_dir: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    info_path = os.path.join(dataset_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cols = list(df.columns)
        cat_cols = [cols[i] for i in info.get("cat_col_idx", []) if i < len(cols)]
        num_cols = [cols[i] for i in info.get("num_col_idx", []) if i < len(cols)]
        listed = set(cat_cols) | set(num_cols)
        for c in cols:
            if c not in listed:
                (num_cols if pd.api.types.is_numeric_dtype(df[c]) else cat_cols).append(c)
        return cat_cols, num_cols
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cat_cols, num_cols


class JointEncoder:
    """z-scores numeric columns, one-hots categorical columns, concatenates."""

    def __init__(self, cat_cols: List[str], num_cols: List[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self._cat_categories: Dict[str, List] = {}
        self._num_mean: np.ndarray = np.array([])
        self._num_std: np.ndarray = np.array([])
        self._cat_slices: Dict[str, Tuple[int, int]] = {}
        self.dim = 0

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        if self.num_cols:
            num = df[self.num_cols].to_numpy(dtype=float)
            self._num_mean = num.mean(axis=0)
            self._num_std = num.std(axis=0) + 1e-6
            blocks.append((num - self._num_mean) / self._num_std)

        offset = len(self.num_cols)
        for c in self.cat_cols:
            cats = sorted(df[c].astype(str).unique().tolist())
            self._cat_categories[c] = cats
            onehot = pd.get_dummies(df[c].astype(str)).reindex(columns=cats, fill_value=0)
            self._cat_slices[c] = (offset, offset + len(cats))
            offset += len(cats)
            blocks.append(onehot.to_numpy(dtype=float))

        matrix = np.concatenate(blocks, axis=1) if blocks else np.empty((len(df), 0))
        self.dim = matrix.shape[1]
        return matrix

    def inverse_transform(self, matrix: np.ndarray) -> pd.DataFrame:
        out = {}
        n_num = len(self.num_cols)
        if n_num:
            num = matrix[:, :n_num] * self._num_std + self._num_mean
            for i, c in enumerate(self.num_cols):
                out[c] = num[:, i]
        for c in self.cat_cols:
            lo, hi = self._cat_slices[c]
            block = matrix[:, lo:hi]
            idx = np.argmax(block, axis=1)
            cats = np.array(self._cat_categories[c])
            out[c] = cats[idx]
        return pd.DataFrame(out, columns=self.num_cols + self.cat_cols)


class DenoiserMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, n_classes: int = 1):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.class_embed = nn.Embedding(max(n_classes, 1), hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)
        y_emb = self.class_embed(y)
        cond = t_emb + y_emb
        return self.net(torch.cat([x, cond], dim=-1))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)
