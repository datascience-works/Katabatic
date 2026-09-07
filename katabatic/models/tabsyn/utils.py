from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# all torch-related code lives here so models.py only imports from utils.py
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =========================
# Config and runtime state
# =========================


@dataclass
class TabSynConfig:
    d_token: int = 4  # paper's value (Appendix G.1)

    decoder_epochs: int = 50
    decoder_batch_size: int = 2048

    diffusion_epochs: int = 500
    diffusion_batch_size: int = 4096
    diffusion_steps: int = 15  # paper recommends <20 for optimal results
    diffusion_hidden_dim: int = 1024  # paper's value

    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    seed: int = 42
    device: str | None = None


@dataclass
class TabSynState:
    # data/meta
    info: dict[str, Any]
    n_num: int
    cat_sizes: list[int]
    cat_encoders: list  # fitted sklearn LabelEncoders, one per categorical col
    token_dim: int
    column_order: list[int]  # numeric->cat->(target at the end)
    scaler_mean: np.ndarray | None  # for numeric inverse
    scaler_std: np.ndarray | None
    # VAE components (Transformer-based, saved as state dicts)
    tokenizer_state: dict  # state_dict for _Tokenizer
    encoder_state: dict  # state_dict for _Encoder
    decoder_state: dict  # state_dict for _Decoder
    # diffusion denoiser
    denoise_fn: nn.Module  # MLPDiffusion
    device: torch.device
    train_rows: int  # default sample count


# ===========
# Utilities
# ===========


def _get_device(pref: str | None) -> torch.device:
    if pref in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_info(data_dir: str) -> dict[str, Any]:
    info_path = os.path.join(data_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)
    # minimal default if missing
    return {"task_type": "binclass", "n_classes": None}


def _load_split_arrays(
    data_dir: str,
    split: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Returns (X_num, X_cat, y) for the split. Any of X_num/X_cat may be None.
    Expects NumPy files like: X_num_train.npy, X_cat_train.npy, y_train.npy
    """

    def _maybe(path: str) -> np.ndarray | None:
        p = os.path.join(data_dir, path)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    Xn = _maybe(f"X_num_{split}.npy")
    Xc = _maybe(f"X_cat_{split}.npy")
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"), allow_pickle=True)
    return Xn, Xc, y


def _concat_xy(
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
    y: np.ndarray,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    For classification, move y to categorical side; for regression, to numerical.
    """
    if task_type == "regression":
        X_num = (
            y.reshape(-1, 1)
            if X_num is None
            else np.concatenate([y.reshape(-1, 1), X_num], axis=1)
        )
        return X_num, X_cat
    # classification-like
    X_cat = (
        y.reshape(-1, 1).astype(str)
        if X_cat is None
        else np.concatenate([y.reshape(-1, 1).astype(str), X_cat], axis=1)
    )
    return X_num, X_cat


def _fit_numeric_scaler(
    X_num: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if X_num is None:
        return None, None
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _transform_numeric(
    X_num: np.ndarray | None, mean: np.ndarray | None, std: np.ndarray | None
) -> np.ndarray | None:
    if X_num is None:
        return None
    return (X_num - mean) / std


def _inverse_numeric(
    X_num: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None
) -> np.ndarray:
    if mean is None or std is None:
        return X_num
    return X_num * std + mean


def _cat_sizes(X_cat: np.ndarray | None) -> list[int]:
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
    X_cat: np.ndarray | None,
) -> tuple[np.ndarray | None, list[LabelEncoder]]:
    """
    Maps each categorical column to integer indices [0..n_classes-1]
    using sklearn LabelEncoder, so encoding is reversible via
    encoder.inverse_transform() later.
    """
    if X_cat is None:
        return None, []
    enc = np.zeros_like(X_cat, dtype=np.int64)
    encoders: list[LabelEncoder] = []
    for j in range(X_cat.shape[1]):
        col = X_cat[:, j].astype(str)
        le = LabelEncoder()
        enc[:, j] = le.fit_transform(col)
        encoders.append(le)
    return enc, encoders


# =======================
# Latent encoder/decoder
# =======================

class _Tokenizer(nn.Module):
    """
    Column-wise tokenizer: converts raw numeric/categorical columns into
    per-column d-dimensional tokens (paper Section 3.2 / Figure 2).
    """

    def __init__(self, n_num: int, cat_sizes: list[int], d_token: int) -> None:
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
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embeds = nn.ModuleList([nn.Embedding(s, d_token) for s in cat_sizes])
        for emb in self.cat_embeds:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(
        self, X_num: torch.Tensor | None, X_cat: torch.Tensor | None
    ) -> torch.Tensor:
        tokens = []
        if X_num is not None and self.n_num > 0:
            num_tok = X_num.unsqueeze(-1) * self.num_weight.unsqueeze(
                0
            ) + self.num_bias.unsqueeze(0)
            tokens.append(num_tok)
        if X_cat is not None and len(self.cat_sizes) > 0:
            cat_tok = [emb(X_cat[:, j]) for j, emb in enumerate(self.cat_embeds)]
            tokens.append(torch.stack(cat_tok, dim=1))
        # (B, M, d_token) -- M = n_num + n_cat, one token per column
        return torch.cat(tokens, dim=1)


class _TransformerBlock(nn.Module):
    """
    Single Transformer layer matching the paper's architecture (Appendix
    D.1): single-head self-attention + 2-layer FFN with ReLU (hidden
    dim D=128), with residual + LayerNorm ("Add & Norm") after each.
    """

    def __init__(self, d_token: int, ffn_hidden: int = 128) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_token),
        )
        self.norm2 = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class _Encoder(nn.Module):
    """
    TabSyn VAE encoder: a 2-layer Transformer (Appendix D.1) producing
    mu and log-sigma for the latent Z, via two parallel Transformer
    branches (mu encoder, log-sigma encoder), matching Figure 7.
    Trainable (not frozen) , this is the key architectural difference
    from the previous simplified implementation.
    """

    def __init__(self, d_token: int, ffn_hidden: int = 128, n_layers: int = 2) -> None:
        super().__init__()
        self.mu_layers = nn.ModuleList(
            [_TransformerBlock(d_token, ffn_hidden) for _ in range(n_layers)]
        )
        self.logsigma_layers = nn.ModuleList(
            [_TransformerBlock(d_token, ffn_hidden) for _ in range(n_layers)]
        )

    def forward(self, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = E
        for layer in self.mu_layers:
            mu = layer(mu)
        logsigma = E
        for layer in self.logsigma_layers:
            logsigma = layer(logsigma)
        return mu, logsigma

class _Decoder(nn.Module):
    """
    TabSyn VAE decoder: a 2-layer Transformer (Appendix D.1), mirroring
    the encoder, that maps latent tokens Z back to (num, cat) column
    predictions.
    """

    def __init__(
        self,
        n_num: int,
        cat_sizes: list[int],
        d_token: int,
        ffn_hidden: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = cat_sizes
        self.d_token = d_token

        self.layers = nn.ModuleList(
            [_TransformerBlock(d_token, ffn_hidden) for _ in range(n_layers)]
        )

        if n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.xavier_uniform_(self.num_weight)
        else:
            self.register_parameter("num_weight", None)

        self.cat_heads = nn.ModuleList([nn.Linear(d_token, s) for s in cat_sizes])
        for head in self.cat_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        """
        z: (B, M, d_token) -- M = n_num + n_cat, one token per column
        """
        tokens = z
        for layer in self.layers:
            tokens = layer(tokens)

        num_pred: torch.Tensor | None = None
        if self.n_num > 0:
            num_toks = tokens[:, : self.n_num]  # (B, n_num, d_token)
            num_pred = (num_toks * self.num_weight.unsqueeze(0)).sum(-1)

        cat_logits: list[torch.Tensor] = []
        for j, head in enumerate(self.cat_heads):
            cat_tok = tokens[:, self.n_num + j]  # (B, d_token)
            cat_logits.append(head(cat_tok))
        return num_pred, cat_logits
    
def _reparameterize(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    """
    Z = mu + sigma * eps, eps ~ N(0, I) (paper Eq. in Section 3.2 / D.1)
    """
    sigma = torch.exp(logsigma)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def _vae_loss(
    pred_num: torch.Tensor | None,
    pred_cat_logits: list[torch.Tensor],
    y_num: torch.Tensor,
    y_cat: torch.Tensor,
    mu: torch.Tensor,
    logsigma: torch.Tensor,
    beta: float,
    n_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = reconstruction loss + beta * KL-divergence.
    Returns (total_loss, recon_loss, kl_loss) for logging.
    """
    recon = torch.tensor(0.0, device=mu.device)
    if n_num > 0 and pred_num is not None and y_num.numel() > 0:
        recon = recon + F.mse_loss(pred_num, y_num)

    if len(pred_cat_logits) > 0 and y_cat.numel() > 0:
        ce = torch.tensor(0.0, device=mu.device)
        for j, logits in enumerate(pred_cat_logits):
            ce = ce + F.cross_entropy(logits, y_cat[:, j])
        recon = recon + ce / max(1, len(pred_cat_logits))

    # KL(N(mu, sigma^2) || N(0, I)), averaged over batch and tokens
    kl = -0.5 * torch.mean(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

    total = recon + beta * kl
    return total, recon, kl


def _scheduled_beta(epoch: int, beta_max: float, beta_min: float, lam: float) -> float:
    """
    Exponential decay schedule for beta (paper Section 4.4 / Table 4):
    beta_t = beta_max * (lam ** epoch), floored at beta_min.
    """
    beta = beta_max * (lam**epoch)
    return max(beta, beta_min)

# ==================
# Diffusion pieces
# ==================


class _PositionalEmbedding(nn.Module):
    def __init__(
        self, num_channels: int, max_positions: int = 10000, endpoint: bool = False
    ):
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
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
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
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        x_in = c_in * x
        F_x = self.denoise_fn(x_in, c_noise.flatten())
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


class _EDMLoss:
    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net: _Precond, data: torch.Tensor) -> torch.Tensor:
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(data) * sigma.unsqueeze(1)
        D_yn = net(data + n, sigma)
        loss = weight.unsqueeze(1) * ((D_yn - data) ** 2)
        return loss.mean()


def _sample_precond(
    net: _Precond,
    num_samples: int,
    dim: int,
    num_steps: int = 50,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    rho = 7
    SIGMA_MIN = max(0.002, net.sigma_min)
    SIGMA_MAX = min(80.0, net.sigma_max)

    latents = torch.randn([num_samples, dim], device=device)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (
        SIGMA_MAX ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (SIGMA_MIN ** (1 / rho) - SIGMA_MAX ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        # churn
        gamma = min(1.0 / num_steps, np.sqrt(2) - 1)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_next + (t_hat**2 - t_cur**2).sqrt() * torch.randn_like(x_next)
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
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[int]]:
    Xn_tr, Xc_tr, y_tr = _load_split_arrays(data_dir, "train")
    task_type = info.get("task_type", "binclass")
    Xn_tr, Xc_tr = _concat_xy(Xn_tr, Xc_tr, y_tr, task_type)

    # Fit numeric scaler
    mean, std = _fit_numeric_scaler(Xn_tr)
    Xn_tr_scaled = _transform_numeric(Xn_tr, mean, std)

# Encode categoricals to indices and collect sizes
    Xc_tr_idx, cat_encoders = _categorical_to_index(Xc_tr)
    sizes = _cat_sizes(Xc_tr)

    return (
        Xn_tr_scaled
        if Xn_tr_scaled is not None
        else np.zeros((len(y_tr), 0), dtype=np.float32),
        Xc_tr_idx if Xc_tr_idx is not None else None,
        y_tr,
        sizes,
        cat_encoders,
        mean,
        std,
    )


def _prepare_split_mats_for_eval(
    data_dir: str,
    info: dict[str, Any],
    mean: np.ndarray | None,
    std: np.ndarray | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
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
    Xn: torch.Tensor | None,
    Xc: torch.Tensor | None,
    batch_size: int,
) -> DataLoader:
    # Prepare labels for decoder training
    y_list: list[torch.Tensor] = []
    if Xn is not None and Xn.numel() > 0:
        y_list.append(Xn.float())
    if Xc is not None:
        y_list.extend([Xc[:, j].long() for j in range(Xc.shape[1])])
    Y = (
        torch.column_stack([t if t.ndim == 2 else t.unsqueeze(1) for t in y_list])
        if y_list
        else torch.zeros(len(z), 0)
    )
    ds = TensorDataset(z, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def train_tabsyn(
    *,
    data_dir: str,
    cfg: TabSynConfig,
    save_dir: str | None = None,
    extra_info: dict[str, Any] = {},
) -> TabSynState:
    _seed_all(cfg.seed)
    device = _get_device(cfg.device)
    info = _load_info(data_dir)

    # ---- Load & prepare training mats
    Xn_tr_np, Xc_tr_idx_np, y_tr, cat_sizes, cat_encoders, mean, std = _prepare_training_mats(data_dir, info)
    n_num = Xn_tr_np.shape[1]
    token_dim = cfg.d_token
    column_order = list(range(n_num)) + list(
        range(n_num, n_num + len(cat_sizes))
    )  # used for DF assembly


    # torch tensors
    Xn_tr = torch.from_numpy(Xn_tr_np).float().to(device) if n_num > 0 else None
    Xc_tr = (
        torch.from_numpy(Xc_tr_idx_np).long().to(device)
        if Xc_tr_idx_np is not None
        else None
    )

       # ---- Tokenize training rows
    tokenizer = _Tokenizer(n_num=n_num, cat_sizes=cat_sizes, d_token=token_dim).to(device)
    E_tr = tokenizer(Xn_tr, Xc_tr)  # (B, M, d_token), M = n_num + n_cat

    # ---- Build trainable VAE encoder + decoder, train jointly
    encoder = _Encoder(d_token=token_dim).to(device)
    decoder = _Decoder(n_num=n_num, cat_sizes=cat_sizes, d_token=token_dim).to(device)

    vae_params = list(tokenizer.parameters()) + list(encoder.parameters()) + list(
        decoder.parameters()
    )
    vae_opt = torch.optim.Adam(vae_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Build labels as single tensors with stable shapes
    y_num_all = (
        Xn_tr
        if (n_num > 0)
        else torch.empty((len(y_tr), 0), device=device, dtype=torch.float32)
    )
    y_cat_all = (
        Xc_tr
        if (Xc_tr is not None)
        else torch.empty((len(y_tr), 0), device=device, dtype=torch.long)
    )

    idx_all = torch.arange(len(y_tr), device=device)
    vae_loader = DataLoader(
        TensorDataset(idx_all, y_num_all, y_cat_all),
        batch_size=cfg.decoder_batch_size,
        shuffle=True,
        num_workers=0,
    )

    beta_max, beta_min, lam = 0.01, 1e-5, 0.7

    for epoch in range(cfg.decoder_epochs):
        tokenizer.train()
        encoder.train()
        decoder.train()
        beta = _scheduled_beta(epoch, beta_max, beta_min, lam)
        total = 0.0
        count = 0
        pbar = tqdm(
            vae_loader,
            desc=f"[vae] epoch {epoch + 1}/{cfg.decoder_epochs} (beta={beta:.5f})",
            leave=False,
        )
        for idx_b, y_num_b, y_cat_b in pbar:
            Xn_b = Xn_tr[idx_b] if Xn_tr is not None else None
            Xc_b = Xc_tr[idx_b] if Xc_tr is not None else None

            E_b = tokenizer(Xn_b, Xc_b)
            mu, logsigma = encoder(E_b)
            z_b = _reparameterize(mu, logsigma)
            pred_num, pred_cat_logits = decoder(z_b)

            loss, recon, kl = _vae_loss(
                pred_num, pred_cat_logits, y_num_b, y_cat_b, mu, logsigma, beta, n_num
            )

            vae_opt.zero_grad(set_to_none=True)
            loss.backward()
            vae_opt.step()

            total += loss.item() * idx_b.size(0)
            count += idx_b.size(0)
            pbar.set_postfix(loss=total / max(1, count), recon=recon.item(), kl=kl.item())

    # ---- Compute final latents z_tr for diffusion training (encoder in eval mode)
    tokenizer.eval()
    encoder.eval()
    with torch.no_grad():
        E_tr = tokenizer(Xn_tr, Xc_tr)
        mu_tr, logsigma_tr = encoder(E_tr)
        z_tr = mu_tr.reshape(mu_tr.shape[0], -1)  # flatten (B, M, d_token) -> (B, M*d_token)
    in_dim = z_tr.shape[1]

    # ---- Train diffusion denoiser on z
    denoise_backbone = MLPDiffusion(d_in=in_dim, dim_t=cfg.diffusion_hidden_dim).to(device)
    precond = _Precond(denoise_backbone, sigma_data=0.5).to(device)
    precond.num_steps = cfg.diffusion_steps
    edm_loss = _EDMLoss()

    dif_loader = DataLoader(
        TensorDataset(z_tr),
        batch_size=cfg.diffusion_batch_size,
        shuffle=True,
        num_workers=0,
    )
    opt = torch.optim.Adam(
        precond.denoise_fn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best = float("inf")
    patience = 0
    for epoch in range(cfg.diffusion_epochs):
        precond.train()
        total = 0.0
        count = 0
        pbar = tqdm(
            dif_loader,
            desc=f"[diffusion] epoch {epoch + 1}/{cfg.diffusion_epochs}",
            leave=False,
        )
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
        cat_encoders=cat_encoders,
        token_dim=token_dim,
        column_order=column_order,
        scaler_mean=mean,
        scaler_std=std,
        tokenizer_state=tokenizer.cpu().state_dict(),
        encoder_state=encoder.cpu().state_dict(),
        decoder_state=decoder.cpu().state_dict(),
        denoise_fn=precond.cpu(),  # keep full precond with backbone
        device=device,
        train_rows=z_tr.shape[0],
    )

    # Optional: save snapshots (single pickle-based bundle for artifact store)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        bundle = {
            "denoise_fn": state.denoise_fn.state_dict(),
            "tokenizer": state.tokenizer_state,
            "encoder": state.encoder_state,
            "decoder": state.decoder_state,
        }
        torch.save(bundle, os.path.join(save_dir, "tabsyn_state.pkl"))
        
    return state

def evaluate_tabsyn(
    state: TabSynState,
    *,
    data_dir: str,
    split: str = "test",
) -> float:
    device = state.device
    info = state.info

    # Restore tokenizer, encoder & decoder modules from saved state dicts
    tokenizer = _Tokenizer(
        n_num=state.n_num, cat_sizes=state.cat_sizes, d_token=state.token_dim
    )
    tokenizer.load_state_dict(state.tokenizer_state)
    tokenizer = tokenizer.to(device)

    encoder = _Encoder(d_token=state.token_dim)
    encoder.load_state_dict(state.encoder_state)
    encoder = encoder.to(device)

    decoder = _Decoder(
        n_num=state.n_num, cat_sizes=state.cat_sizes, d_token=state.token_dim
    )
    decoder.load_state_dict(state.decoder_state)
    decoder = decoder.to(device)

    Xn, Xc, _ = _prepare_split_mats_for_eval(
        data_dir, info, state.scaler_mean, state.scaler_std, split
    )
    Xn_t = torch.from_numpy(Xn).float().to(device) if state.n_num > 0 else None
    Xc_t = torch.from_numpy(Xc).long().to(device) if Xc is not None else None

    with torch.no_grad():
        E = tokenizer(Xn_t, Xc_t)
        mu, _ = encoder(E)
        pred_num, pred_cat_logits = decoder(mu)

        loss = 0.0
        if (
            state.n_num > 0
            and pred_num is not None
            and Xn_t is not None
            and Xn_t.numel() > 0
        ):
            loss += F.mse_loss(pred_num, Xn_t).item()
        for j, logits in enumerate(pred_cat_logits):
            loss += F.cross_entropy(logits, Xc_t[:, j]).item()
        denom = (1 if (state.n_num > 0) else 0) + len(pred_cat_logits)
        return loss / max(1, denom)


def _rebuild_decoder_from_state(state: TabSynState) -> _Decoder:
    dec = _Decoder(state.n_num, state.cat_sizes, state.token_dim)
    dec.load_state_dict(state.decoder_state)
    return dec


def _rebuild_encoder_from_state(
    state: TabSynState, device: torch.device | None = None
) -> _Encoder:
    enc = _Encoder(d_token=state.token_dim)
    enc.load_state_dict(state.encoder_state)
    if device is not None:
        enc = enc.to(device)
    return enc


def _rebuild_tokenizer_from_state(
    state: TabSynState, device: torch.device | None = None
) -> _Tokenizer:
    tok = _Tokenizer(
        n_num=state.n_num, cat_sizes=state.cat_sizes, d_token=state.token_dim
    )
    tok.load_state_dict(state.tokenizer_state)
    if device is not None:
        tok = tok.to(device)
    return tok

def sample_tabsyn(
    state: TabSynState,
    *,
    n_samples: int | None = None,
    return_df: bool = True,
) -> pd.DataFrame | np.ndarray:
    device = state.device

    denoise = state.denoise_fn.to(device).eval()
    decoder = _rebuild_decoder_from_state(state).to(device).eval()

    # in_dim: one token per column (no CLS token), flattened
    n_cols = state.n_num + len(state.cat_sizes)
    in_dim = n_cols * state.token_dim

    n = n_samples or state.train_rows
    with torch.no_grad():
        num_steps = getattr(
            state.denoise_fn, "num_steps", 50
        )  # default to 50 if absent
        z_flat = _sample_precond(denoise, n, in_dim, num_steps=num_steps, device=device)
        z = z_flat.view(n, n_cols, state.token_dim)
        # decode
        pred_num, pred_cat_logits = decoder(z)
        

        # numerics inverse scale
        Xn_hat = None
        if state.n_num > 0 and pred_num is not None:
            Xn_hat = pred_num.cpu().numpy()
            Xn_hat = _inverse_numeric(Xn_hat, state.scaler_mean, state.scaler_std)

        # categoricals argmax, then decode back to original string labels
        Xc_hat = None
        if len(state.cat_sizes):
            Xc_logits = [log.cpu().numpy() for log in pred_cat_logits]
            Xc_hat_idx = np.stack([logits.argmax(axis=1) for logits in Xc_logits], axis=1)

            Xc_hat = np.empty(Xc_hat_idx.shape, dtype=object)
            for j, encoder in enumerate(state.cat_encoders):
                Xc_hat[:, j] = encoder.inverse_transform(Xc_hat_idx[:, j])

    # Build output
    if not return_df:
        if Xn_hat is None and Xc_hat is None:
            return np.zeros((n, 0), dtype=np.float32)
        if Xn_hat is None:
            return Xc_hat
        if Xc_hat is None:
            return Xn_hat
        return np.concatenate([Xn_hat, Xc_hat], axis=1)

    # As DataFrame
    cols: list[str] = []
    if state.n_num > 0:
        cols += [f"num_{i}" for i in range(state.n_num)]
    cols += [f"cat_{i}" for i in range(len(state.cat_sizes))]
    df = pd.DataFrame(
        np.concatenate(
            [
                Xn_hat if Xn_hat is not None else np.zeros((n, 0)),
                Xc_hat if Xc_hat is not None else np.zeros((n, 0)),
            ],
            axis=1,
        ),
        columns=cols,
    )

    idx_name = state.info.get("idx_name_mapping")
    if isinstance(idx_name, dict):
        idx_name = {int(k): v for k, v in idx_name.items()}
        new_names = []
        for i, c in enumerate(cols):
            new_names.append(idx_name.get(i, c))
        df.columns = new_names

    return df