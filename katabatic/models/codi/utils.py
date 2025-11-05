"""
Utility functions for the CoDi model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import math
import random
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder


def infer_schema(df: pd.DataFrame, categorical_threshold: int = 20) -> Dict[str, Any]:
    """Infer schema from DataFrame."""
    schema = {
        'continuous_columns': [],
        'categorical_columns': [],
        'column_info': {},
        'column_order': list(df.columns)
    }

    for col in df.columns:
        col_data = df[col].dropna()
        n_unique = col_data.nunique()

        if pd.api.types.is_numeric_dtype(col_data):
            if n_unique <= categorical_threshold:
                schema['categorical_columns'].append(col)
                unique_vals = sorted(col_data.unique())
                schema['column_info'][col] = {
                    'type': 'categorical',
                    'size': len(unique_vals),
                    'values': unique_vals
                }
            else:
                schema['continuous_columns'].append(col)
                schema['column_info'][col] = {
                    'type': 'continuous',
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std())
                }
        else:
            schema['categorical_columns'].append(col)
            unique_vals = sorted(col_data.unique())
            schema['column_info'][col] = {
                'type': 'categorical',
                'size': len(unique_vals),
                'values': unique_vals
            }

    return schema


def encode_dataframe(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode DataFrame according to schema."""
    df_encoded = df.copy()

    for col in schema['continuous_columns']:
        col_info = schema['column_info'][col]
        col_data = df_encoded[col]
        col_min = col_info['min']
        col_max = col_info['max']

        if col_max > col_min:
            normalized = 2 * (col_data - col_min) / (col_max - col_min) - 1
        else:
            normalized = col_data * 0

        df_encoded[col] = normalized.astype(np.float32)

    for col in schema['categorical_columns']:
        col_info = schema['column_info'][col]
        encoder = LabelEncoder()
        encoder.classes_ = np.array(col_info['values'])
        df_encoded[col] = encoder.transform(df_encoded[col]).astype(np.float32)
        col_info['encoder'] = encoder

    return df_encoded, schema


def decode_dataframe(df_encoded: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """Decode DataFrame back to original format."""
    df_decoded = df_encoded.copy()

    for col in schema['continuous_columns']:
        col_info = schema['column_info'][col]
        col_min = col_info['min']
        col_max = col_info['max']

        if col_max > col_min:
            denormalized = (df_decoded[col] + 1) * \
                (col_max - col_min) / 2 + col_min
        else:
            denormalized = df_decoded[col] * 0 + col_min

        df_decoded[col] = denormalized

    for col in schema['categorical_columns']:
        col_info = schema['column_info'][col]
        col_data_int = np.clip(
            np.round(df_decoded[col]), 0, col_info['size'] - 1).astype(int)

        if 'encoder' in col_info:
            df_decoded[col] = col_info['encoder'].inverse_transform(
                col_data_int)
        else:
            df_decoded[col] = col_data_int

    return df_decoded


def save_metadata(schema: Dict[str, Any], filepath: str):
    """Save schema metadata to JSON."""
    schema_clean = {
        'continuous_columns': schema['continuous_columns'],
        'categorical_columns': schema['categorical_columns'],
        'column_order': schema['column_order'],
        'column_info': {}
    }

    for col, info in schema['column_info'].items():
        info_clean = {k: v for k, v in info.items() if k != 'encoder'}
        if 'values' in info_clean:
            info_clean['values'] = [str(v) for v in info_clean['values']]
        schema_clean['column_info'][col] = info_clean

    with open(filepath, 'w') as f:
        json.dump(schema_clean, f, indent=2)


def load_metadata(filepath: str) -> Dict[str, Any]:
    """Load schema metadata from JSON."""
    with open(filepath, 'r') as f:
        schema = json.load(f)

    for col in schema['categorical_columns']:
        col_info = schema['column_info'][col]
        if 'values' in col_info:
            encoder = LabelEncoder()
            encoder.classes_ = np.array(col_info['values'])
            col_info['encoder'] = encoder

    return schema


def set_global_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device."""
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                    device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


def extract(v: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from v at indices t."""
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def default_init():
    """Default initialization for neural networks."""
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return _init


def get_activation(activation: str) -> nn.Module:
    """Get activation function."""
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'silu':
        return nn.SiLU()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU()


class TabularUNet(nn.Module):
    """U-Net for tabular diffusion."""

    def __init__(self, input_dim, cond_dim, output_dim, hidden_dims, time_embed_dim, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim
        self.activation = get_activation(activation)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            self.activation,
            nn.Linear(time_embed_dim * 4, time_embed_dim * 4)
        )

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, max(input_dim // 2, 1))

        # Input layer
        self.input_layer = nn.Linear(
            input_dim + max(input_dim // 2, 1), hidden_dims[0])

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder.append(nn.ModuleList([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.Linear(time_embed_dim * 4, hidden_dims[i + 1]),
                self.activation
            ]))

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder.append(nn.ModuleList([
                nn.Linear(hidden_dims[i] * 2, hidden_dims[i - 1]),
                nn.Linear(time_embed_dim * 4, hidden_dims[i - 1]),
                self.activation
            ]))

        # Output
        self.output_layer = nn.Linear(hidden_dims[0], output_dim)

        self.apply(default_init())

    def forward(self, x, t, cond):
        """Forward pass."""
        t_emb = get_timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        cond_proj = self.cond_proj(cond)
        h = torch.cat([x, cond_proj], dim=1)
        h = self.input_layer(h)
        h = self.activation(h)

        skips = []
        for layer1, temb_proj, act in self.encoder:
            h = layer1(h)
            h = act(h + temb_proj(t_emb))
            skips.append(h)

        for (layer1, temb_proj, act), skip in zip(self.decoder, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = layer1(h)
            h = act(h + temb_proj(t_emb))

        return self.output_layer(h)


class GaussianDiffusionTrainer(nn.Module):
    """Gaussian diffusion trainer."""

    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             torch.sqrt(1 - alphas_bar))

    def forward(self, x_0, t, cond):
        """Training step."""
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        noise_pred = self.model(x_t, t, cond)
        return F.mse_loss(noise_pred, noise)


class GaussianDiffusionSampler(nn.Module):
    """Gaussian diffusion sampler."""

    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        posterior_variance = betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        posterior_log_variance = torch.log(
            torch.clamp(posterior_variance, min=1e-20))

        posterior_mean_coef1 = betas * \
            torch.sqrt(alphas_bar_prev) / (1 - alphas_bar)
        posterior_mean_coef2 = (1 - alphas_bar_prev) * \
            torch.sqrt(alphas) / (1 - alphas_bar)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             torch.sqrt(1 - alphas_bar))
        self.register_buffer('sqrt_recip_alphas_bar',
                             torch.sqrt(1.0 / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar',
                             torch.sqrt(1.0 / alphas_bar - 1))
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('posterior_log_variance', posterior_log_variance)

    def p_sample(self, x_t, t, cond):
        """Single sampling step."""
        eps = self.model(x_t, t, cond)
        x_0_pred = (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_variance = extract(self.posterior_log_variance, t, x_t.shape)

        noise = torch.randn_like(x_t)
        noise[t == 0] = 0

        return mean + torch.exp(0.5 * log_variance) * noise

    def forward(self, x_T, cond):
        """Generate samples."""
        x_t = x_T
        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full(
                (x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, cond)
        return x_t


class MultinomialDiffusion(nn.Module):
    """Multinomial diffusion for categorical features."""

    def __init__(self, model, num_classes, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas_bar', alphas_bar)

    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - add noise to clean data.

        Args:
            log_x_start: Log probabilities of clean one-hot data
            t: Timesteps

        Returns:
            Noisy log probabilities
        """
        alphas_bar_t = extract(self.alphas_bar, t, log_x_start.shape)

        # For categorical: interpolate between one-hot and uniform
        x_start = torch.exp(log_x_start)
        uniform_prob = torch.ones_like(x_start) / x_start.shape[1]
        noisy_prob = alphas_bar_t * x_start + (1 - alphas_bar_t) * uniform_prob

        # Sample with Gumbel-softmax
        gumbel = - \
            torch.log(-torch.log(torch.rand_like(noisy_prob) + 1e-20) + 1e-20)
        x_t = F.softmax((noisy_prob.log() + gumbel) / 0.1, dim=-1)

        return torch.log(x_t.clamp(min=1e-30))

    def compute_Lt(self, log_x_start: torch.Tensor, log_x_t: torch.Tensor,
                   t: torch.Tensor, cond: torch.Tensor) -> tuple:
        """
        Compute the loss for timestep t.

        Args:
            log_x_start: Log of clean one-hot data
            log_x_t: Log of noisy one-hot data
            t: Timesteps
            cond: Conditional features

        Returns:
            Tuple of (loss, predicted log x_0)
        """
        # Predict x_0 from x_t
        logits = self.model(torch.exp(log_x_t), t, cond)
        log_x_0_pred = F.log_softmax(logits, dim=-1)

        # Compute KL divergence
        kl = F.kl_div(log_x_0_pred, torch.exp(log_x_start),
                      reduction='none').sum(dim=-1)

        return kl, log_x_0_pred

    def kl_prior(self, log_x_start: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence with prior.

        Args:
            log_x_start: Log probabilities of clean data

        Returns:
            KL divergence
        """
        # Prior is uniform distribution
        uniform_log_prob = -torch.log(torch.tensor(float(log_x_start.shape[1]),
                                                   device=log_x_start.device))
        uniform_log_prob = uniform_log_prob * torch.ones_like(log_x_start)

        kl = F.kl_div(uniform_log_prob, torch.exp(
            log_x_start), reduction='none').sum(dim=-1)
        return kl

    def forward(self, x_0_onehot, t, cond):
        """Training step."""
        alphas_bar_t = extract(self.alphas_bar, t, x_0_onehot.shape)

        # Create noisy one-hot
        uniform_prob = torch.ones_like(x_0_onehot) / x_0_onehot.shape[1]
        noisy_prob = alphas_bar_t * x_0_onehot + \
            (1 - alphas_bar_t) * uniform_prob

        # Sample with Gumbel-softmax
        gumbel = - \
            torch.log(-torch.log(torch.rand_like(noisy_prob) + 1e-20) + 1e-20)
        x_t = F.softmax((noisy_prob.log() + gumbel) / 0.1, dim=-1)

        # Predict
        logits = self.model(x_t, t, cond)

        return F.binary_cross_entropy_with_logits(logits, x_0_onehot)

    def p_sample(self, x_t, t, cond):
        """Single sampling step."""
        logits = self.model(x_t, t, cond)

        if t[0] == 0:
            return F.softmax(logits, dim=-1)

        # Sample with Gumbel
        gumbel = - \
            torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        return F.softmax((logits + gumbel) / 0.1, dim=-1)

    def reverse_sample(self, x_T, cond):
        """Generate samples."""
        x_t = x_T
        for t in range(self.T - 1, -1, -1):
            t_batch = torch.full(
                (x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, cond)
        return x_t
