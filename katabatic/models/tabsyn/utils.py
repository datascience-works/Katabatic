from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import os
import numpy as np
import pandas as pd

# all torch-related code lives here so models.py only imports from utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================
# Config and runtime state
# =========================

@dataclass
class TabSynConfig:
    d_token: int = 16

    decoder_epochs: int = 50
    decoder_batch_size: int = 2048

    diffusion_epochs: int = 500
    diffusion_batch_size: int = 4096
    diffusion_steps: int = 50

    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    seed: int = 42
    device: Optional[str] = None


@dataclass
class TabSynState:
    # data/meta
    info: Dict[str, Any]
    n_num: int
    cat_sizes: List[int]
    token_dim: int
    column_order: List[int]               # numeric->cat->(target at the end)
    scaler_mean: Optional[np.ndarray]     # for numeric inverse
    scaler_std: Optional[np.ndarray]
    # encoder/decoder
    encoder_num_weight: torch.Tensor      # (n_num, d_token)
    encoder_num_bias: torch.Tensor        # (n_num, d_token)
    encoder_cat_embeds: List[torch.Tensor]  # per categorical col: (n_classes, d_token)
    decoder_num_weight: torch.Tensor      # (n_num, d_token)
    decoder_cat_heads: List[nn.Linear]    # per categorical col
    # diffusion denoiser
    denoise_fn: nn.Module                 # MLPDiffusion
    device: torch.device
    train_rows: int                       # default sample count


# ===========
# Utilities
# ===========

def _get_device(pref: Optional[str]) -> torch.device:
    if pref in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_info(data_dir: str) -> Dict[str, Any]:
    info_path = os.path.join(data_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)
    # minimal default if missing
    return {"task_type": "binclass", "n_classes": None}


def _load_split_arrays(
    data_dir: str,
    split: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Returns (X_num, X_cat, y) for the split. Any of X_num/X_cat may be None.
    Expects NumPy files like: X_num_train.npy, X_cat_train.npy, y_train.npy
    """
    def _maybe(path: str) -> Optional[np.ndarray]:
        p = os.path.join(data_dir, path)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    Xn = _maybe(f"X_num_{split}.npy")
    Xc = _maybe(f"X_cat_{split}.npy")
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"), allow_pickle=True)
    return Xn, Xc, y


def _concat_xy(
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    y: np.ndarray,
    task_type: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    For classification, move y to categorical side; for regression, to numerical.
    """
    if task_type == "regression":
        X_num = y.reshape(-1, 1) if X_num is None else np.concatenate([y.reshape(-1, 1), X_num], axis=1)
        return X_num, X_cat
    # classification-like
    X_cat = y.reshape(-1, 1).astype(str) if X_cat is None else np.concatenate([y.reshape(-1, 1).astype(str), X_cat], axis=1)
    return X_num, X_cat


def _fit_numeric_scaler(X_num: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if X_num is None:
        return None, None
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _transform_numeric(X_num: Optional[np.ndarray], mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if X_num is None:
        return None
    return (X_num - mean) / std


def _inverse_numeric(X_num: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return X_num
    return X_num * std + mean


def _cat_sizes(X_cat: Optional[np.ndarray]) -> List[int]:
    if X_cat is None:
        return []
    sizes = []
    for j in range(X_cat.shape[1]):
        col = X_cat[:, j]
        # robust: treat as strings and enumerate unique values
        _, inv = np.unique(col.astype(str), return_inverse=True)
        sizes.append(int(inv.max()) + 1)
    return sizes


def _categorical_to_index(
    X_cat: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], List[Dict[str, int]]]:
    """
    Maps each categorical column to integer indices [0..n_classes-1].
    Returns encoded array and per-column vocab dicts.
    """
    if X_cat is None:
        return None, []
    enc = np.zeros_like(X_cat, dtype=np.int64)
    vocabs: List[Dict[str, int]] = []
    for j in range(X_cat.shape[1]):
        col = X_cat[:, j].astype(str)
        uniq = np.unique(col)
        vocab = {v: i for i, v in enumerate(uniq)}
        enc[:, j] = np.vectorize(vocab.get)(col)
        vocabs.append(vocab)
    return enc, vocabs


# =======================
# Latent encoder/decoder
# =======================

class _Encoder(nn.Module):
    """
    Simple deterministic encoder producing token embeddings:
      - one [CLS]-like dummy token (zeros),
      - one token per numeric feature: v * W_j + b_j,
      - one token per categorical feature: Embedding(category_id).
    """
    def __init__(self, n_num: int, cat_sizes: List[int], d_token: int) -> None:
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = cat_sizes
        self.d_token = d_token

        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=np.sqrt(5))
            nn.init.uniform_(self.num_bias, -0.01, 0.01)
        else:
            # dummy params
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(s, d_token) for s in cat_sizes]
        )
        for emb in self.cat_embeds:
            nn.init.normal_(emb.weight, std=0.02)

        # encoder is deterministic & fixed during training of decoder/diffusion
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, X_num: Optional[torch.Tensor], X_cat: Optional[torch.Tensor]) -> torch.Tensor:
        B = (X_num.shape[0] if X_num is not None else X_cat.shape[0]) if (X_num is not None or X_cat is not None) else 0
        tokens = [torch.zeros(B, 1, self.d_token, device=X_num.device if X_num is not None else X_cat.device)]
        if X_num is not None and self.n_num > 0:
            # (B, n_num, d_token)
            num_tok = X_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tok)
        if X_cat is not None and len(self.cat_sizes) > 0:
            # list of (B, d_token) -> (B, n_cat, d_token)
            cat_tok = [emb(X_cat[:, j]) for j, emb in enumerate(self.cat_embeds)]
            tokens.append(torch.stack(cat_tok, dim=1))
        # (B, n_tokens, d_token) -> flatten
        T = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        return T.reshape(B, -1)


class _Decoder(nn.Module):
    """
    Lightweight decoder from flattened tokens back to (num, cat) columns.
    For each numeric feature, a linear head over its token.
    For each categorical feature, a linear head producing class logits.
    """
    def __init__(self, n_num: int, cat_sizes: List[int], d_token: int) -> None:
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = cat_sizes
        self.d_token = d_token

        # One linear "readout vector" per numeric feature
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.xavier_uniform_(self.num_weight)
        else:
            self.register_parameter("num_weight", None)

        # One classifier per categorical feature
        self.cat_heads = nn.ModuleList([nn.Linear(d_token, s) for s in cat_sizes])
        for head in self.cat_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        z: (B, (1 + n_num + n_cat) * d_token)
        """
        B = z.shape[0]
        n_cat = len(self.cat_sizes)
        n_tokens = 1 + self.n_num + n_cat
        # (B, n_tokens, d_token)
        tokens = z.view(B, n_tokens, self.d_token)

        # numeric tokens follow CLS
        num_pred: Optional[torch.Tensor] = None
        if self.n_num > 0:
            num_toks = tokens[:, 1:1 + self.n_num]                    # (B, n_num, d_token)
            # (B, n_num)
            num_pred = (num_toks * self.num_weight.unsqueeze(0)).sum(-1)

        # categorical tokens follow numeric
        cat_logits: List[torch.Tensor] = []
        for j, head in enumerate(self.cat_heads):
            cat_tok = tokens[:, 1 + self.n_num + j]                   # (B, d_token)
            cat_logits.append(head(cat_tok))                          # (B, n_classes)
        return num_pred, cat_logits


# ==================
# Diffusion pieces
# ==================

class _PositionalEmbedding(nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = torch.outer(x, freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in: int, dim_t: int = 512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = _PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        # noise bounds (compatible with sampling below)
        self.sigma_min = 0.002
        self.sigma_max = 80.0

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        return self.mlp(x)


class _Precond(nn.Module):
    def __init__(self, denoise_fn: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.sigma_min = 0.0
        self.sigma_max = float("inf")
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_in = c_in * x
        F_x = self.denoise_fn(x_in, c_noise.flatten())
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


class _EDMLoss:
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net: _Precond, data: torch.Tensor) -> torch.Tensor:
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(data) * sigma.unsqueeze(1)
        D_yn = net(data + n, sigma)
        loss = weight.unsqueeze(1) * ((D_yn - data) ** 2)
        return loss.mean()


def _sample_precond(
    net: _Precond,
    num_samples: int,
    dim: int,
    num_steps: int = 50,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    rho = 7
    SIGMA_MIN = max(0.002, net.sigma_min)
    SIGMA_MAX = min(80.0, net.sigma_max)

    latents = torch.randn([num_samples, dim], device=device)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (SIGMA_MAX ** (1 / rho) + step_indices / (num_steps - 1) * (SIGMA_MIN ** (1 / rho) - SIGMA_MAX ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # churn
        gamma = min(1.0 / num_steps, np.sqrt(2) - 1)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_next + (t_hat ** 2 - t_cur ** 2).sqrt() * torch.randn_like(x_next)
        # euler
        den = net(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - den) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # 2nd order
        if i < num_steps - 1:
            den2 = net(x_next, t_next).to(torch.float32)
            d_prime = (x_next - den2) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


# ==========================
# Training / Evaluation API
# ==========================

def _prepare_training_mats(
    data_dir: str,
    info: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[int]]:
    Xn_tr, Xc_tr, y_tr = _load_split_arrays(data_dir, "train")
    task_type = info.get("task_type", "binclass")
    Xn_tr, Xc_tr = _concat_xy(Xn_tr, Xc_tr, y_tr, task_type)

    # Fit numeric scaler
    mean, std = _fit_numeric_scaler(Xn_tr)
    Xn_tr_scaled = _transform_numeric(Xn_tr, mean, std)

    # Encode categoricals to indices and collect sizes
    Xc_tr_idx, vocabs = _categorical_to_index(Xc_tr)
    sizes = _cat_sizes(Xc_tr)

    return (
        Xn_tr_scaled if Xn_tr_scaled is not None else np.zeros((len(y_tr), 0), dtype=np.float32),
        Xc_tr_idx if Xc_tr_idx is not None else None,
        y_tr,
        sizes,
    )


def _prepare_split_mats_for_eval(
    data_dir: str,
    info: Dict[str, Any],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    split: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    Xn, Xc, y = _load_split_arrays(data_dir, split)
    task_type = info.get("task_type", "binclass")
    Xn, Xc = _concat_xy(Xn, Xc, y, task_type)
    Xn_scaled = _transform_numeric(Xn, mean, std) if Xn is not None else None
    return (
        Xn_scaled if Xn_scaled is not None else np.zeros((len(y), 0), dtype=np.float32),
        _categorical_to_index(Xc)[0] if Xc is not None else None,
        y,
    )


def _make_dataloaders_from_latent(
    z: torch.Tensor,
    Xn: Optional[torch.Tensor],
    Xc: Optional[torch.Tensor],
    batch_size: int,
) -> DataLoader:
    # Prepare labels for decoder training
    y_list: List[torch.Tensor] = []
    if Xn is not None and Xn.numel() > 0:
        y_list.append(Xn.float())
    if Xc is not None:
        y_list.extend([Xc[:, j].long() for j in range(Xc.shape[1])])
    Y = torch.column_stack([t if t.ndim == 2 else t.unsqueeze(1) for t in y_list]) if y_list else torch.zeros(len(z), 0)
    ds = TensorDataset(z, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def train_tabsyn(
    *,
    data_dir: str,
    cfg: TabSynConfig,
    save_dir: Optional[str] = None,
    extra_info: Dict[str, Any] = {},
) -> TabSynState:
    _seed_all(cfg.seed)
    device = _get_device(cfg.device)
    info = _load_info(data_dir)

    # ---- Load & prepare training mats
    Xn_tr_np, Xc_tr_idx_np, y_tr, cat_sizes = _prepare_training_mats(data_dir, info)
    n_num = Xn_tr_np.shape[1]
    token_dim = cfg.d_token
    column_order = list(range(n_num)) + list(range(n_num, n_num + len(cat_sizes)))  # used for DF assembly

    # scalers for inverse-transform
    mean, std = _fit_numeric_scaler(None if n_num == 0 else (Xn_tr_np * 1.0))

    # torch tensors
    Xn_tr = torch.from_numpy(Xn_tr_np).float().to(device) if n_num > 0 else None
    Xc_tr = torch.from_numpy(Xc_tr_idx_np).long().to(device) if Xc_tr_idx_np is not None else None

    # ---- Build encoder (frozen), create latents z for training rows
    encoder = _Encoder(n_num=n_num, cat_sizes=cat_sizes, d_token=token_dim).to(device)
    with torch.no_grad():
        z_tr = encoder(Xn_tr, Xc_tr)  # (B, in_dim)
    in_dim = z_tr.shape[1]

    # ---- Train decoder over z -> (num, cat)
    decoder = _Decoder(n_num=n_num, cat_sizes=cat_sizes, d_token=token_dim).to(device)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

   # Build labels as single tensors with stable shapes
    y_num_all = Xn_tr if (n_num > 0) else torch.empty((len(z_tr), 0), device=device, dtype=torch.float32)
    y_cat_all = (
        Xc_tr if (Xc_tr is not None) else torch.empty((len(z_tr), 0), device=device, dtype=torch.long)
    )

    dec_loader = DataLoader(
        TensorDataset(z_tr, y_num_all, y_cat_all),
        batch_size=cfg.decoder_batch_size,
        shuffle=True,
        num_workers=0,
    )



    for epoch in range(cfg.decoder_epochs):
        decoder.train()
        total = 0.0
        count = 0
        pbar = tqdm(dec_loader, desc=f"[decoder] epoch {epoch+1}/{cfg.decoder_epochs}", leave=False)
        for z_b, y_num_b, y_cat_b in pbar:
            pred_num, pred_cat_logits = decoder(z_b)
            loss = 0.0

            if n_num > 0 and pred_num is not None and y_num_b.numel() > 0:
                loss = loss + F.mse_loss(pred_num, y_num_b)

            if len(cat_sizes) > 0 and len(pred_cat_logits) > 0 and y_cat_b.numel() > 0:
                assert y_cat_b.dtype in (torch.int64, torch.long)
                assert y_cat_b.ndim == 2, f"expected (B, n_cat), got {y_cat_b.shape}"
                ce = 0.0
                for j, logits in enumerate(pred_cat_logits):
                    ce = ce + F.cross_entropy(logits, y_cat_b[:, j])
                loss = loss + ce / max(1, len(pred_cat_logits))

            # ✅ add these lines:
            dec_opt.zero_grad(set_to_none=True)
            loss.backward()
            dec_opt.step()

            total += loss.item() * z_b.size(0)
            count += z_b.size(0)
            pbar.set_postfix(loss=total / max(1, count))


    # ---- Train diffusion denoiser on z
    denoise_backbone = MLPDiffusion(d_in=in_dim, dim_t=512).to(device)
    precond = _Precond(denoise_backbone, sigma_data=0.5).to(device)
    precond.num_steps = cfg.diffusion_steps
    edm_loss = _EDMLoss()

    dif_loader = DataLoader(TensorDataset(z_tr), batch_size=cfg.diffusion_batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(precond.denoise_fn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    patience = 0
    for epoch in range(cfg.diffusion_epochs):
        precond.train()
        total = 0.0
        count = 0
        pbar = tqdm(dif_loader, desc=f"[diffusion] epoch {epoch+1}/{cfg.diffusion_epochs}", leave=False)
        for (z_b,) in pbar:
            loss = edm_loss(precond, z_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * z_b.size(0)
            count += z_b.size(0)
            pbar.set_postfix(loss=total / count)

        avg = total / max(1, count)
        if avg < best:
            best = avg
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    # ---- Assemble state
    state = TabSynState(
        info=info,
        n_num=n_num,
        cat_sizes=cat_sizes,
        token_dim=token_dim,
        column_order=column_order,
        scaler_mean=mean,
        scaler_std=std,
        encoder_num_weight=encoder.num_weight.detach().cpu() if encoder.num_weight is not None else torch.empty(0),
        encoder_num_bias=encoder.num_bias.detach().cpu() if encoder.num_bias is not None else torch.empty(0),
        encoder_cat_embeds=[emb.weight.detach().cpu() for emb in encoder.cat_embeds],
        decoder_num_weight=decoder.num_weight.detach().cpu() if decoder.num_weight is not None else torch.empty(0),
        decoder_cat_heads=[head.cpu() for head in decoder.cat_heads],
        denoise_fn=precond.cpu(),   # keep full precond with backbone
        device=device,
        train_rows=z_tr.shape[0],
    )

    # Optional: save snapshots
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state.denoise_fn.state_dict(), os.path.join(save_dir, "tabsyn_denoiser.pt"))
        torch.save({"num_weight": state.decoder_num_weight, "cat": [h.state_dict() for h in state.decoder_cat_heads]},
                   os.path.join(save_dir, "tabsyn_decoder.pt"))

    return state


def evaluate_tabsyn(
    state: TabSynState,
    *,
    data_dir: str,
    split: str = "test",
) -> float:
    device = state.device
    info = state.info

    # Restore encoder & decoder modules from state
    encoder = _Encoder(n_num=state.n_num, cat_sizes=state.cat_sizes, d_token=state.token_dim)
    if state.n_num > 0:
        encoder.num_weight.data = state.encoder_num_weight.clone()
        encoder.num_bias.data = state.encoder_num_bias.clone()
    for emb, W in zip(encoder.cat_embeds, state.encoder_cat_embeds):
        emb.weight.data = W.clone()
    encoder = encoder.to(device)

    decoder = _Decoder(n_num=state.n_num, cat_sizes=state.cat_sizes, d_token=state.token_dim).to(device)
    if state.n_num > 0:
        decoder.num_weight.data = state.decoder_num_weight.clone()
    for head, sd in zip(decoder.cat_heads, [h.state_dict() for h in state.decoder_cat_heads]):
        head.load_state_dict(sd)

    Xn, Xc, _ = _prepare_split_mats_for_eval(
        data_dir, info, state.scaler_mean, state.scaler_std, split
    )
    Xn_t = torch.from_numpy(Xn).float().to(device) if state.n_num > 0 else None
    Xc_t = torch.from_numpy(Xc).long().to(device) if Xc is not None else None

    with torch.no_grad():
        z = encoder(Xn_t, Xc_t)
        pred_num, pred_cat_logits = decoder(z)

        loss = 0.0
        if state.n_num > 0 and pred_num is not None and Xn_t is not None and Xn_t.numel() > 0:
            loss += F.mse_loss(pred_num, Xn_t).item()
        for j, logits in enumerate(pred_cat_logits):
            loss += F.cross_entropy(logits, Xc_t[:, j]).item()
        # Normalize by number of heads (optional)
        denom = (1 if (state.n_num > 0) else 0) + len(pred_cat_logits)
        return loss / max(1, denom)


def _rebuild_decoder_from_state(state: TabSynState) -> _Decoder:
    dec = _Decoder(state.n_num, state.cat_sizes, state.token_dim)
    if state.n_num > 0:
        dec.num_weight.data = state.decoder_num_weight.clone()
    for head, saved in zip(dec.cat_heads, [h.state_dict() for h in state.decoder_cat_heads]):
        head.load_state_dict(saved)
    return dec


def _rebuild_encoder_from_state(state: TabSynState, device: Optional[torch.device] = None) -> _Encoder:
    enc = _Encoder(state.n_num, state.cat_sizes, state.token_dim)
    if state.n_num > 0:
        enc.num_weight.data = state.encoder_num_weight.clone()
        enc.num_bias.data = state.encoder_num_bias.clone()
    for emb, W in zip(enc.cat_embeds, state.encoder_cat_embeds):
        emb.weight.data = W.clone()
    if device is not None:
        enc = enc.to(device)
    return enc


def sample_tabsyn(
    state: TabSynState,
    *,
    n_samples: Optional[int] = None,
    return_df: bool = True,
) -> 'pd.DataFrame | np.ndarray':
    device = state.device

    denoise = state.denoise_fn.to(device).eval()
    decoder = _rebuild_decoder_from_state(state).to(device).eval()

    # Use encoder to get in_dim
    enc = _rebuild_encoder_from_state(state, device=device).eval()
    # fabricate a fake batch to infer in_dim
    in_dim = (1 + state.n_num + len(state.cat_sizes)) * state.token_dim

    n = n_samples or state.train_rows
    with torch.no_grad():
        num_steps = getattr(state.denoise_fn, "num_steps", 50)  # default to 50 if absent
        z = _sample_precond(denoise, n, in_dim, num_steps=num_steps, device=device)
        # center around typical scale (mean-zeroed)
        # decode
        pred_num, pred_cat_logits = decoder(z)

        # numerics inverse scale
        Xn_hat = None
        if state.n_num > 0 and pred_num is not None:
            Xn_hat = pred_num.cpu().numpy()
            Xn_hat = _inverse_numeric(Xn_hat, state.scaler_mean, state.scaler_std)

        # categoricals argmax
        Xc_hat = None
        if len(state.cat_sizes):
            Xc_logits = [log.cpu().numpy() for log in pred_cat_logits]
            Xc_hat = np.stack([logits.argmax(axis=1) for logits in Xc_logits], axis=1)

    # Build output
    if not return_df:
        # return concatenated numeric + categorical indices
        if Xn_hat is None and Xc_hat is None:
            return np.zeros((n, 0), dtype=np.float32)
        if Xn_hat is None:
            return Xc_hat
        if Xc_hat is None:
            return Xn_hat
        return np.concatenate([Xn_hat, Xc_hat], axis=1)

    # As DataFrame
    cols: List[str] = []
    if state.n_num > 0:
        cols += [f"num_{i}" for i in range(state.n_num)]
    cols += [f"cat_{i}" for i in range(len(state.cat_sizes))]
    df = pd.DataFrame(
        np.concatenate(
            [Xn_hat if Xn_hat is not None else np.zeros((n, 0)), Xc_hat if Xc_hat is not None else np.zeros((n, 0))]
            , axis=1
        ),
        columns=cols,
    )

    # Optional: rename using info mapping, if present
    idx_name = state.info.get("idx_name_mapping")
    if isinstance(idx_name, dict):
        idx_name = {int(k): v for k, v in idx_name.items()}
        # try to map in order: numeric first then categorical
        new_names = []
        for i, c in enumerate(cols):
            # raw index attempt; fallback to current name
            new_names.append(idx_name.get(i, c))
        df.columns = new_names

    return df
