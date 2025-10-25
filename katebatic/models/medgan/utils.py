"""
MedGAN Neural Network Components and Utilities.

Based on "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks"
by Choi et al. (2017) - https://arxiv.org/abs/1703.06490
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class Autoencoder(nn.Module):
    """
    Autoencoder for compressing high-dimensional binary/count data.
    
    Architecture:
        Encoder: input_dim -> encoder_dim -> latent_dim
        Decoder: latent_dim -> encoder_dim -> input_dim
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        bn_decay: float = 0.99
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim, momentum=1-bn_decay),
            nn.Tanh(),
            nn.Linear(encoder_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=1-bn_decay)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim, momentum=1-bn_decay),
            nn.Tanh(),
            nn.Linear(encoder_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return torch.sigmoid(self.decoder(z))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class Generator(nn.Module):
    """
    Generator network that produces synthetic data in the latent space.
    
    Architecture:
        noise (latent_dim) -> hidden -> hidden -> output (latent_dim)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bn_decay: float = 0.99
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        in_dim = latent_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=1-bn_decay),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate latent representation from random noise."""
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish real from synthetic data.
    
    Architecture:
        input (latent_dim) -> hidden -> hidden -> output (1)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict probability that input is real (vs synthetic)."""
        return torch.sigmoid(self.model(z))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = 'rbf', 
                bandwidth: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    Args:
        x: Samples from first distribution [n, d]
        y: Samples from second distribution [m, d]
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: Bandwidth parameter for RBF kernel
    
    Returns:
        MMD value
    """
    def rbf_kernel(x, y, bandwidth):
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        
        x_sqnorms = torch.diag(xx)
        y_sqnorms = torch.diag(yy)
        
        k_xx = torch.exp(-(x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx) / (2 * bandwidth ** 2))
        k_yy = torch.exp(-(y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy) / (2 * bandwidth ** 2))
        k_xy = torch.exp(-(x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy) / (2 * bandwidth ** 2))
        
        return k_xx, k_yy, k_xy
    
    if kernel == 'rbf':
        k_xx, k_yy, k_xy = rbf_kernel(x, y, bandwidth)
    else:  # linear kernel
        k_xx = torch.mm(x, x.t())
        k_yy = torch.mm(y, y.t())
        k_xy = torch.mm(x, y.t())
    
    m, n = x.shape[0], y.shape[0]
    mmd = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    
    return mmd


def sample_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Sample random noise from normal distribution."""
    return torch.randn(batch_size, latent_dim, device=device)
