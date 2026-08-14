"""Utility classes for the Katabatic Flow-VAE model.

This file contains the PyTorch normalizing-flow VAE components and tabular
pre/post-processing helpers used by models.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def m(t: torch.Tensor) -> torch.Tensor:
            return F.softplus(t) - 1.0

        def h(t: torch.Tensor) -> torch.Tensor:
            return torch.tanh(t)

        def h_prime(t: torch.Tensor) -> torch.Tensor:
            return 1.0 - h(t).pow(2)

        inner = (self.w * self.u).sum()
        u_hat = self.u + (m(inner) - inner) * self.w / (self.w.norm().pow(2) + 1e-8)
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        z = x + u_hat * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1.0 + (u_hat * psi).sum(dim=1, keepdim=True)) + 1e-8)
        return z, log_det


class RadialFlow(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.d = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.exp(self.a)
        b = -a + F.softplus(self.b)
        r = (x - self.c).norm(dim=1, keepdim=True)
        h = 1.0 / (a + r + 1e-8)
        h_prime = -h.pow(2)
        tmp = b * h
        z = x + tmp * (x - self.c)
        log_det = (self.d - 1) * torch.log(torch.abs(1.0 + tmp) + 1e-8) + torch.log(
            torch.abs(1.0 + tmp + b * h_prime * r) + 1e-8
        )
        return z, log_det


class HouseholderFlow(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.v = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.d = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outer = self.v.t() @ self.v
        v_sqr = self.v.norm().pow(2) + 1e-8
        eye = torch.eye(self.d, device=x.device, dtype=x.dtype)
        H = eye - 2.0 * outer / v_sqr
        z = x @ H.t()
        log_det = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        return z, log_det


class NiceFlow(nn.Module):
    def __init__(self, dim: int, mask: int, final: bool = False) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("NICE flow requires an even latent_dim.")
        self.final = final
        if final:
            self.scale = nn.Parameter(torch.zeros(1, dim))
        else:
            self.mask = mask
            self.coupling = nn.Sequential(
                nn.Linear(dim // 2, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim * 5),
                nn.ReLU(),
                nn.Linear(dim * 5, dim // 2),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.final:
            z = x * torch.exp(self.scale)
            log_det = torch.sum(self.scale).expand(x.size(0), 1)
            return z, log_det

        bsz, width = x.size()
        x_pair = x.reshape(bsz, width // 2, 2)
        if self.mask:
            on, off = x_pair[:, :, 0], x_pair[:, :, 1]
        else:
            off, on = x_pair[:, :, 0], x_pair[:, :, 1]
        on = on + self.coupling(off)
        if self.mask:
            z = torch.stack((on, off), dim=2)
        else:
            z = torch.stack((off, on), dim=2)
        log_det = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
        return z.reshape(bsz, width), log_det


class Flow(nn.Module):
    def __init__(self, dim: int, flow_type: str = "none", length: int = 0) -> None:
        super().__init__()
        flow_type = (flow_type or "none").lower()
        if flow_type == "planar":
            modules = [PlanarFlow(dim) for _ in range(length)]
        elif flow_type == "radial":
            modules = [RadialFlow(dim) for _ in range(length)]
        elif flow_type == "householder":
            modules = [HouseholderFlow(dim) for _ in range(length)]
        elif flow_type == "nice":
            modules = [NiceFlow(dim, i % 2, i == length - 1) for i in range(length)]
        else:
            modules = []
        self.flow = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for module in self.flow:
            x, inc = module(x)
            log_det = log_det + inc
        return x, log_det


class GatedLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.gate(x)


class MLPLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gate: bool) -> None:
        super().__init__()
        self.layer = GatedLayer(in_dim, out_dim) if gate else nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FlowVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        layers: int = 2,
        gate: bool = False,
        flow_type: str = "planar",
        flow_length: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.ModuleList(
            [MLPLayer(in_dim, hidden_dim, gate)]
            + [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(max(0, layers - 1))]
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.flow = Flow(latent_dim, flow_type, flow_length)
        self.decoder = nn.ModuleList(
            [MLPLayer(latent_dim, hidden_dim, gate)]
            + [MLPLayer(hidden_dim, hidden_dim, gate) for _ in range(max(0, layers - 1))]
            + [nn.Linear(hidden_dim, in_dim)]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.encoder:
            x = layer(x)
        return self.mean(x), torch.clamp(self.log_var(x), min=-8.0, max=8.0)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0 = eps.mul(std).add(mean)
        zk, log_det = self.flow(z0)
        return zk, log_det

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        z, log_det = self.reparameterize(mean, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mean, log_var, log_det

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, mean, log_var, log_det = self.forward(x)
        recon_loss = F.mse_loss(reconstruction, x, reduction="none").sum(dim=1, keepdim=True)
        kl = -0.5 * torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp(), dim=1, keepdim=True)
        return (recon_loss + kl - log_det).mean()

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


@dataclass
class TabularSchema:
    columns: List[str]
    feature_columns: List[str]
    label_column: str
    numeric_columns: List[str]
    categorical_columns: List[str]
    encoded_columns: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    categories: Dict[str, List[Any]]


def fit_transform_tabular(
    df: pd.DataFrame,
    label_column: str | None = None,
    categorical_cols: List[str] | None = None,
    continuous_cols: List[str] | None = None,
) -> Tuple[np.ndarray, TabularSchema]:
    """Encode a mixed-type dataframe into a numeric matrix for VAE training."""
    df = df.copy()
    if label_column is None:
        label_column = df.columns[-1]

    columns = df.columns.tolist()
    feature_columns = [c for c in columns if c != label_column]

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    categories: Dict[str, List[Any]] = {}
    parts: List[pd.DataFrame] = []
    encoded_columns: List[str] = []

    for col in columns:
        series = df[col]
        if categorical_cols is not None and col in categorical_cols:
            is_numeric = False
        elif continuous_cols is not None and col in continuous_cols:
            is_numeric = True
        else:
            is_numeric = pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10
        if is_numeric:
            numeric_columns.append(col)
            values = pd.to_numeric(series, errors="coerce")
            mean = float(values.mean()) if not np.isnan(values.mean()) else 0.0
            std = float(values.std()) if values.std() and not np.isnan(values.std()) else 1.0
            means[col] = mean
            stds[col] = std
            encoded = ((values.fillna(mean) - mean) / std).to_frame(col)
            parts.append(encoded)
            encoded_columns.append(col)
        else:
            categorical_columns.append(col)
            filled = series.astype("object").where(series.notna(), "__MISSING__").astype(str)
            cats = sorted(filled.unique().tolist())
            categories[col] = cats
            one_hot = pd.get_dummies(filled, prefix=col, prefix_sep="__", dtype=float)
            expected_cols = [f"{col}__{cat}" for cat in cats]
            one_hot = one_hot.reindex(columns=expected_cols, fill_value=0.0)
            parts.append(one_hot)
            encoded_columns.extend(expected_cols)

    encoded_df = pd.concat(parts, axis=1).astype(float)
    schema = TabularSchema(
        columns=columns,
        feature_columns=feature_columns,
        label_column=label_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        encoded_columns=encoded_columns,
        means=means,
        stds=stds,
        categories=categories,
    )
    return encoded_df.to_numpy(dtype=np.float32), schema


def inverse_transform_tabular(array: np.ndarray, schema: TabularSchema) -> pd.DataFrame:
    """Decode a generated numeric matrix back into the original dataframe schema."""
    array = np.asarray(array)
    data: Dict[str, Any] = {}
    idx = 0
    for col in schema.columns:
        if col in schema.numeric_columns:
            values = array[:, idx] * schema.stds[col] + schema.means[col]
            data[col] = values
            idx += 1
        else:
            cats = schema.categories[col]
            width = len(cats)
            block = array[:, idx : idx + width]
            chosen = np.argmax(block, axis=1)
            data[col] = [cats[i] for i in chosen]
            idx += width
    return pd.DataFrame(data, columns=schema.columns)
