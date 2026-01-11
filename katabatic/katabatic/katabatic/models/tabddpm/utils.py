"""Utility classes for the Tabddpm model (local fallback).

This module provides lightweight stand-ins for the external `tabddpm` package
so that the Tabddpm model can run without that dependency. The implementations
below are intentionally simple and optimized for stability rather than fidelity.
They provide the minimal interface expected by `models.py`:

- MLPDiffusion: a simple MLP denoiser conditioned on class labels
- GaussianMultinomialDiffusion: a wrapper exposing `mixed_loss` and `sample_all`

If the real `tabddpm` package is installed, `models.py` will prefer that.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDiffusion(nn.Module):
    """Simple MLP denoiser.

    Parameters
    ----------
    d_in : int
        Input dimensionality (numerical + categorical indices concatenated).
    num_classes : int
        Number of classes for optional y-conditioning (0 disables conditioning).
    is_y_cond : bool
        Whether to condition the denoiser on y labels.
    rtdl_params : dict
        Contains 'd_layers' (list of hidden widths) and 'dropout' (float).
    dim_t : int
        Unused here; accepted for API compatibility.
    """

    def __init__(
        self,
        d_in: int,
        num_classes: int,
        is_y_cond: bool,
        rtdl_params: dict,
        dim_t: int = 128,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.is_y_cond = bool(is_y_cond)
        self.num_classes = int(num_classes) if num_classes else 0

        hidden = list(rtdl_params.get("d_layers", [256, 256]))
        dropout = float(rtdl_params.get("dropout", 0.0))

        cond_dim = self.num_classes if self.is_y_cond and self.num_classes > 0 else 0
        layers: list[nn.Module] = []
        in_dim = self.d_in + cond_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_y_cond and self.num_classes > 0 and y is not None:
            # Ensure y is long for one-hot
            y_onehot = F.one_hot(
                y.long().view(-1), num_classes=self.num_classes).float()
            x = torch.cat([x, y_onehot], dim=1)
        return self.net(x)


class GaussianMultinomialDiffusion(nn.Module):
    """Minimal diffusion wrapper with mixed loss and sampling.

    This class holds a denoiser and exposes:
    - mixed_loss(xb, out) -> (loss_multi, loss_gauss)
    - sample_all(num_samples, batch_size, y_dist, ddim=False) -> (x, y)
    """

    def __init__(
        self,
        num_classes: Sequence[int] | np.ndarray,
        num_numerical_features: int,
        denoise_fn: nn.Module,
        gaussian_loss_type: str = "mse",
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._denoise_fn = denoise_fn
        self.register_buffer("_dummy", torch.zeros(1))  # for .to(device)

        # bookkeeping
        self._K = np.array(num_classes, dtype=int) if len(
            num_classes) > 0 else np.array([0], dtype=int)
        self._n_num = int(num_numerical_features)
        self._n_cat = int(np.sum(self._K > 0))
        self._gauss_loss = gaussian_loss_type
        self._steps = int(num_timesteps)
        self._scheduler = scheduler

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._denoise_fn(x, y)

    def mixed_loss(self, xb: torch.Tensor, out: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a simple reconstruction-style mixed loss.

        We split the loss into two parts for reporting parity:
        - loss_multi: on categorical dimensions (L1)
        - loss_gauss: on numerical dimensions (MSE)
        """
        y = out.get("y") if isinstance(out, dict) else None
        x_pred = self._denoise_fn(xb, y)

        d_total = xb.shape[1]
        n_num = min(self._n_num, d_total)
        n_cat = max(d_total - n_num, 0)

        loss_gauss = torch.tensor(0.0, device=xb.device)
        loss_multi = torch.tensor(0.0, device=xb.device)

        if n_num > 0:
            x_num = xb[:, :n_num]
            x_num_pred = x_pred[:, :n_num]
            loss_gauss = F.mse_loss(x_num_pred, x_num)

        if n_cat > 0:
            x_cat = xb[:, n_num:]
            x_cat_pred = x_pred[:, n_num:]
            # L1 over categorical indices (rounded) to stay stable
            loss_multi = F.l1_loss(torch.round(x_cat_pred), torch.round(x_cat))

        return loss_multi, loss_gauss

    @torch.no_grad()
    def sample_all(
        self,
        num_samples: int,
        *,
        batch_size: int,
        y_dist: torch.Tensor,
        ddim: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic x and y.

        We draw y ~ Categorical(y_dist) if provided, otherwise zeros. Then we
        generate x by sampling simple noise and passing it through the denoiser
        once (single-step), followed by clamping/rounding for stability.
        """
        device = self._dummy.device
        n = int(num_samples)

        # sample y
        if y_dist is not None and y_dist.numel() > 1:
            y_dist = (y_dist / y_dist.sum()).to(device)
            classes = torch.arange(len(y_dist), device=device)
            y_samples = torch.multinomial(y_dist, n, replacement=True)
            y_vec = classes[y_samples]
        else:
            y_vec = torch.zeros(n, dtype=torch.long, device=device)

        d_total = self._n_num + self._n_cat
        if d_total == 0:
            d_total = 1

        xs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bsz = end - start
            # base noise
            x0 = torch.randn(bsz, self._n_num,
                             device=device) if self._n_num > 0 else None
            if self._n_cat > 0:
                # uniform categorical indices per column
                cat_cols = []
                for k in self._K[self._K > 0]:
                    idx = torch.randint(low=0, high=int(
                        k), size=(bsz, 1), device=device).float()
                    cat_cols.append(idx)
                xcat = torch.cat(cat_cols, dim=1) if cat_cols else None
            else:
                xcat = None

            if x0 is None and xcat is None:
                x_in = torch.zeros(bsz, d_total, device=device)
            elif x0 is None:
                x_in = xcat
            elif xcat is None:
                x_in = x0
            else:
                x_in = torch.cat([x0, xcat], dim=1)

            y_batch = y_vec[start:end]
            x_out = self._denoise_fn(x_in, y_batch)
            # numeric clamp, categorical round
            if self._n_num > 0:
                x_out[:, :self._n_num] = x_out[:,
                                               :self._n_num].clamp_(-3.0, 3.0)
            if self._n_cat > 0:
                x_out[:, self._n_num:] = torch.round(
                    x_out[:, self._n_num:]).clamp_(min=0)
            xs.append(x_out)

        X = torch.cat(xs, dim=0)[:n]
        return X, y_vec[:n]

# Add your utility functions here.
