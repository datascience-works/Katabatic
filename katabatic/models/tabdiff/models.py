"""Implementation of TabDiff: joint diffusion over numeric + categorical columns.

Kotelnikov et al.'s TabDDPM (already in this repo, see katabatic/models/tabddpm)
runs two separate diffusion processes — Gaussian for numeric columns,
multinomial for categorical columns. TabDiff's distinguishing idea is a single,
*unified* diffusion process over the joint feature space
(Shi et al., 2024/ICLR 2025, https://arxiv.org/abs/2410.20626).

This implementation follows that unified-process idea directly: categorical
columns are one-hot encoded and numeric columns z-scored into one continuous
vector, a single Gaussian DDPM diffuses/denoises that joint vector, and
categorical blocks are recovered with argmax at sampling time. It is a
simplified version of the paper (the paper additionally learns per-feature-type
adaptive noise schedules and uses a more elaborate transformer denoiser) — see
README.md "Status" for the gap between this and the full paper method.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from katabatic.models.base_model import Model
from .utils import JointEncoder, DenoiserMLP, cosine_beta_schedule, load_column_roles


class Tabdiff(Model):
    _defaults = dict(
        steps=300,
        num_timesteps=100,
        lr=1e-3,
        batch_size=128,
        hidden=128,
        seed=42,
    )

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.cfg = {**self._defaults, **(config or {})}
        self.device = torch.device("cpu")
        self._encoder: Optional[JointEncoder] = None
        self._net: Optional[DenoiserMLP] = None
        self._betas: Optional[torch.Tensor] = None
        self._y_encoder: Optional[LabelEncoder] = None
        self._y_col: Optional[str] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch"]

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        config: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> "Tabdiff":
        if config:
            self.cfg.update(config)
        torch.manual_seed(self.cfg["seed"])

        X = pd.read_csv(os.path.join(dataset_dir, "x_train.csv"))
        y = pd.read_csv(os.path.join(dataset_dir, "y_train.csv"))
        self._y_col = y.columns[0]

        cat_cols, num_cols = load_column_roles(dataset_dir, X)
        self._encoder = JointEncoder(cat_cols, num_cols)
        matrix = self._encoder.fit_transform(X)
        x0 = torch.tensor(matrix, dtype=torch.float32, device=self.device)

        self._y_encoder = LabelEncoder()
        y_idx = torch.tensor(
            self._y_encoder.fit_transform(y[self._y_col].astype(str)),
            dtype=torch.long,
            device=self.device,
        )
        n_classes = len(self._y_encoder.classes_)

        T = self.cfg["num_timesteps"]
        self._betas = cosine_beta_schedule(T).to(self.device)
        alphas = 1.0 - self._betas
        self._alphas_cumprod = torch.cumprod(alphas, dim=0)

        self._net = DenoiserMLP(
            dim=self._encoder.dim, hidden=self.cfg["hidden"], n_classes=n_classes
        ).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg["lr"])

        n = x0.shape[0]
        bsz = min(self.cfg["batch_size"], n)
        for step in range(self.cfg["steps"]):
            idx = torch.randint(0, n, (bsz,))
            x_batch = x0[idx]
            y_batch = y_idx[idx]
            t = torch.randint(0, T, (bsz,), device=self.device)
            noise = torch.randn_like(x_batch)
            a_bar = self._alphas_cumprod[t].unsqueeze(-1)
            x_t = torch.sqrt(a_bar) * x_batch + torch.sqrt(1 - a_bar) * noise

            pred_noise = self._net(x_t, t, y_batch)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % max(1, self.cfg["steps"] // 5) == 0:
                print(f"[tabdiff] step {step + 1}/{self.cfg['steps']} loss={loss.item():.4f}")

        self._n_classes = n_classes
        self._class_probs = torch.bincount(y_idx, minlength=n_classes).float()
        self._class_probs /= self._class_probs.sum()
        self.is_fitted = True

        if synthetic_dir is not None:
            df_synth = self.sample(n)
            os.makedirs(synthetic_dir, exist_ok=True)
            x_synth = df_synth[self._encoder.num_cols + self._encoder.cat_cols]
            y_synth = df_synth[[self._y_col]]
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    @torch.no_grad()
    def sample(self, n: int, *args, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        T = self.cfg["num_timesteps"]
        betas = self._betas
        alphas = 1.0 - betas
        alphas_cumprod = self._alphas_cumprod

        y_idx = torch.multinomial(self._class_probs, n, replacement=True)
        x = torch.randn(n, self._encoder.dim, device=self.device)

        for t_step in reversed(range(T)):
            t = torch.full((n,), t_step, dtype=torch.long, device=self.device)
            pred_noise = self._net(x, t, y_idx)
            a_t = alphas[t_step]
            a_bar_t = alphas_cumprod[t_step]
            beta_t = betas[t_step]

            mean = (1 / torch.sqrt(a_t)) * (
                x - (beta_t / torch.sqrt(1 - a_bar_t)) * pred_noise
            )
            if t_step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        matrix = x.cpu().numpy()
        df = self._encoder.inverse_transform(matrix)
        df[self._y_col] = self._y_encoder.inverse_transform(y_idx.cpu().numpy())
        return df
