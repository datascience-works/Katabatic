"""TVAE-GAN: rebuilt to match Larsen et al. (2016), 'Autoencoding beyond
pixels using a learned similarity metric' (arXiv:1512.09300).

Architecture: a single shared Decoder/Generator network, a VAE-style
Encoder, and a Discriminator, trained jointly with the paper's three-part
loss (KL prior, discriminator-feature reconstruction, adversarial loss).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TabularPreprocessor:
    """
    Converts a real-world tabular DataFrame into a single float array
    suitable for neural network training, and reverses that conversion
    for generated output.

    Numeric columns are standardised (mean 0, std 1). Categorical columns
    (including any explicitly forced target columns) are one-hot encoded.
    """

    categorical_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    _encoder: OneHotEncoder | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _col_order: list[str] = field(default_factory=list, init=False, repr=False)
    _dtypes: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def infer_and_fit(
        cls, df: pd.DataFrame, force_categorical: list[str] | None = None, max_unique: int = 20
    ) -> "TabularPreprocessor":
        force_categorical = set(force_categorical or [])
        cat_cols, num_cols = [], []
        for col in df.columns:
            if col in force_categorical or df[col].dtype == object or df[col].nunique() <= max_unique:
                cat_cols.append(col)
            else:
                num_cols.append(col)

        prep = cls(categorical_cols=cat_cols, numeric_cols=num_cols)
        prep._col_order = list(df.columns)
        prep._dtypes = df.dtypes.to_dict()

        if cat_cols:
            prep._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            prep._encoder.fit(df[cat_cols].astype(str))
        if num_cols:
            prep._scaler = StandardScaler()
            prep._scaler.fit(df[num_cols])

        return prep

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.categorical_cols:
            parts.append(self._encoder.transform(df[self.categorical_cols].astype(str)))
        if self.numeric_cols:
            parts.append(self._scaler.transform(df[self.numeric_cols]))
        if not parts:
            raise ValueError("TabularPreprocessor found no usable columns.")
        return np.concatenate(parts, axis=1).astype(np.float32)

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        cat_width = (
            sum(len(c) for c in self._encoder.categories_) if self.categorical_cols else 0
        )
        cat_part, num_part = arr[:, :cat_width], arr[:, cat_width:]

        frames = []
        if self.categorical_cols:
            decoded_cat = self._encoder.inverse_transform(cat_part)
            frames.append(pd.DataFrame(decoded_cat, columns=self.categorical_cols))
        if self.numeric_cols:
            decoded_num = self._scaler.inverse_transform(num_part)
            frames.append(pd.DataFrame(decoded_num, columns=self.numeric_cols))

        out = pd.concat(frames, axis=1)[self._col_order]
        for col, dtype in self._dtypes.items():
            if np.issubdtype(dtype, np.integer):
                out[col] = out[col].round().astype(dtype)
            elif np.issubdtype(dtype, np.floating):
                out[col] = out[col].astype(dtype)
            else:
                out[col] = out[col].astype(dtype)
        return out


class _MLP(torch.nn.Module):
    """A simple multi-layer perceptron block, reused by all three networks."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Linear(dims[-1], out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(torch.nn.Module):
    """
    VAE encoder (paper Eq. 1): maps data x to latent distribution
    parameters (mu, logsigma). Sampling uses the reparameterization
    trick, z = mu + sigma * eps.
    """

    def __init__(self, data_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        self.body = _MLP(data_dim, hidden_dims, hidden_dims[-1] if hidden_dims else data_dim)
        feat_dim = hidden_dims[-1] if hidden_dims else data_dim
        self.to_mu = torch.nn.Linear(feat_dim, latent_dim)
        self.to_logsigma = torch.nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.body(x))
        return self.to_mu(h), self.to_logsigma(h)


class DecoderGenerator(torch.nn.Module):
    """
    The paper's shared Decoder/Generator (Fig. 1): the SAME network is
    used both to reconstruct real data from its latent code, and to
    generate new data from a randomly sampled latent code. There is
    only ever one instance of this class per model, used in both roles.
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int], data_dim: int) -> None:
        super().__init__()
        self.net = _MLP(latent_dim, hidden_dims, data_dim)
        self.out_activation = torch.nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out_activation(self.net(z))
    
class Discriminator(torch.nn.Module):
    """
    GAN discriminator (paper Section 2.2), with the hidden-layer feature
    extraction the paper's reconstruction loss depends on (Eq. 6-7).

    Unlike a standard discriminator, this exposes an intermediate layer's
    activations via `features()`, not just the final real/fake score.
    """

    def __init__(self, data_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [data_dim] + hidden_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(
                torch.nn.Sequential(torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.LeakyReLU(0.2))
            )
        self.blocks = torch.nn.ModuleList(blocks)
        self.classifier = torch.nn.Linear(dims[-1], 1)
        self.feature_layer_idx = len(self.blocks) // 2

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Dis_l(x): the hidden representation at self.feature_layer_idx."""
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == self.feature_layer_idx:
                return h
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the real/fake probability, sigmoid output (paper uses BCE)."""
        h = x
        for block in self.blocks:
            h = block(h)
        return torch.sigmoid(self.classifier(h))
    
def reparameterize(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    """z = mu + sigma * eps, eps ~ N(0, I) (paper Eq. 1)."""
    sigma = torch.exp(logsigma)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def kl_prior_loss(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    """L_prior = D_KL(q(z|x) || p(z)), p(z) = N(0, I) (paper Eq. 4)."""
    return -0.5 * torch.mean(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())


def discriminator_feature_loss(
    disc: "Discriminator", x_real: torch.Tensor, x_recon: torch.Tensor
) -> torch.Tensor:
    """
    L_dis_like: reconstruction error measured in the discriminator's
    hidden-layer feature space, not raw data space (paper Eq. 6-7).
    Uses detached discriminator features as a fixed similarity metric,
    the discriminator itself is not updated by this loss (see training
    loop: this loss only flows into Encoder/DecoderGenerator parameters).
    """
    feat_real = disc.features(x_real)
    feat_recon = disc.features(x_recon)
    return F.mse_loss(feat_recon, feat_real)


def gan_loss(
    disc: "Discriminator", x_real: torch.Tensor, x_recon: torch.Tensor, x_prior: torch.Tensor
) -> torch.Tensor:
    """
    L_GAN (paper Eq. 10): standard binary cross-entropy adversarial loss,
    using real samples, reconstructions, and prior-sampled generations.
    """
    real_pred = disc(x_real)
    recon_pred = disc(x_recon)
    prior_pred = disc(x_prior)

    real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    recon_loss = F.binary_cross_entropy(recon_pred, torch.zeros_like(recon_pred))
    prior_loss = F.binary_cross_entropy(prior_pred, torch.zeros_like(prior_pred))
    return real_loss + recon_loss + prior_loss

@dataclass
class TVAEGANConfig:
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    epochs: int = 50
    batch_size: int = 64          # paper's value
    lr: float = 3e-4              # paper's value (0.0003)
    gamma: float = 1.0            # paper's Dec weighting (Eq. 9); not given a
                                   # specific value in the paper's main text,
                                   # 1.0 is an inferred, documented default
    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_vaegan(
    data: np.ndarray, cfg: TVAEGANConfig
) -> tuple[Encoder, DecoderGenerator, Discriminator]:
    """
    Implements Algorithm 1 from the paper: jointly trains Encoder,
    DecoderGenerator, and Discriminator with the correct gradient
    isolation rules:
      - Discriminator's own parameters are updated ONLY by L_GAN
        (never by L_dis_like, which would collapse it to 0).
      - Encoder's parameters are updated ONLY by L_prior + L_dis_like
        (L_GAN's gradient is NOT backpropagated into Encoder).
      - DecoderGenerator's parameters are updated by a weighted
        combination of L_dis_like (reconstruction) and L_GAN (fooling
        the discriminator), weighted by gamma (Eq. 9).
    """
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = data.shape[1]

    encoder = Encoder(data_dim, cfg.hidden_dims, cfg.latent_dim).to(device)
    decoder_generator = DecoderGenerator(cfg.latent_dim, list(reversed(cfg.hidden_dims)), data_dim).to(device)
    discriminator = Discriminator(data_dim, cfg.hidden_dims).to(device)

    # Paper: single RMSProp learning rate for all networks (Section 4)
    enc_opt = torch.optim.RMSprop(encoder.parameters(), lr=cfg.lr)
    dec_opt = torch.optim.RMSprop(decoder_generator.parameters(), lr=cfg.lr)
    dis_opt = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.lr)

    tensor_data = torch.from_numpy(data).float().to(device)
    n = tensor_data.shape[0]

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        epoch_losses = {"prior": 0.0, "dis_like": 0.0, "gan": 0.0}
        n_batches = 0

        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            x = tensor_data[idx]
            batch_size = x.shape[0]

            
            mu, logsigma = encoder(x)
            z = reparameterize(mu, logsigma)
            x_recon = decoder_generator(z)

            z_prior = torch.randn(batch_size, cfg.latent_dim, device=device)
            x_prior = decoder_generator(z_prior)

            
            enc_opt.zero_grad()
            l_prior = kl_prior_loss(mu, logsigma)
            l_dis_like_enc = discriminator_feature_loss(discriminator, x, x_recon)
            encoder_loss = l_prior + l_dis_like_enc
            encoder_loss.backward(retain_graph=True)
            enc_opt.step()

            
            dec_opt.zero_grad()
            # Recompute reconstruction/prior with fresh graph tied to decoder
            mu_d, logsigma_d = encoder(x)
            z_d = reparameterize(mu_d, logsigma_d)
            x_recon_d = decoder_generator(z_d)
            x_prior_d = decoder_generator(torch.randn(batch_size, cfg.latent_dim, device=device))

            l_dis_like_dec = discriminator_feature_loss(discriminator, x, x_recon_d)
            l_gan_dec = gan_loss(discriminator, x, x_recon_d, x_prior_d)
            decoder_loss = (cfg.gamma * l_dis_like_dec) - l_gan_dec
            decoder_loss.backward(retain_graph=True)
            dec_opt.step()

            
            dis_opt.zero_grad()
            x_recon_dis = decoder_generator(z.detach())
            x_prior_dis = decoder_generator(z_prior.detach())
            l_gan_dis = gan_loss(discriminator, x, x_recon_dis.detach(), x_prior_dis.detach())
            l_gan_dis.backward()
            dis_opt.step()

            epoch_losses["prior"] += l_prior.item()
            epoch_losses["dis_like"] += l_dis_like_dec.item()
            epoch_losses["gan"] += l_gan_dis.item()
            n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
        print(
            f"[tvaegan] epoch {epoch + 1}/{cfg.epochs} "
            f"prior={avg['prior']:.4f} dis_like={avg['dis_like']:.4f} gan={avg['gan']:.4f}"
        )

    return encoder, decoder_generator, discriminator