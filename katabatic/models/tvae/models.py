"""Katabatic-compatible self-contained TVAE-style model.

This version intentionally removes the SDV dependency. It implements a compact
PyTorch variational autoencoder for mixed tabular data and keeps the same
Katabatic interface:

    TVAEModel().train(data_dir, synthetic_dir="...")

The model trains on x_train.csv/y_train.csv and writes x_synth.csv/y_synth.csv.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    TabularPreprocessor,
    combine_features_target,
    infer_categorical_columns,
    load_train_split,
    save_synthetic_outputs,
    split_synthetic,
)


class _TabularVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class TVAEModel:
    """Self-contained TVAE-style generator for Katabatic's 3-file format.

    Parameters are deliberately modest by default so the model can run inside
    notebook/coursework environments without SDV.
    """

    def __init__(
        self,
        categorical_columns: Optional[Iterable[str]] = None,
        sample_size: Optional[int] = None,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        random_state: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.categorical_columns = list(categorical_columns) if categorical_columns else None
        self.sample_size = sample_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[_TabularVAE] = None
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.target_col: Optional[str] = None
        self.feature_columns: Optional[list[str]] = None
        self.n_train_rows: Optional[int] = None

    def train(self, data_dir: str, *args, **kwargs):
        """Train on Katabatic split files and optionally save synthetic data.

        Args:
            data_dir: Directory containing x_train.csv and y_train.csv.
            synthetic_dir: Optional output directory for x_synth.csv/y_synth.csv.
            sample_size: Optional synthetic row count override.
            categorical_columns: Optional manual categorical column list.
        """
        synthetic_dir = kwargs.get("synthetic_dir")
        sample_size = kwargs.get("sample_size", self.sample_size)
        categorical_columns = kwargs.get("categorical_columns", self.categorical_columns)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_train, y_train, target_col = load_train_split(data_dir)
        train_df, target_col = combine_features_target(x_train, y_train, target_col)

        self.target_col = target_col
        self.feature_columns = list(x_train.columns)
        self.n_train_rows = len(train_df)

        if categorical_columns is None:
            categorical_columns = infer_categorical_columns(train_df, target_col)
        else:
            categorical_columns = list(categorical_columns)
            if target_col not in categorical_columns:
                categorical_columns.append(target_col)

        self.preprocessor = TabularPreprocessor(categorical_columns=categorical_columns).fit(train_df)
        encoded = self.preprocessor.transform(train_df)

        dataset = TensorDataset(torch.tensor(encoded, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.model = _TabularVAE(
            input_dim=self.preprocessor.output_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, mu, logvar = self.model(batch)

                recon_loss = nn.functional.mse_loss(recon, batch, reduction="mean")
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl_loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        if synthetic_dir is not None:
            self.generate(synthetic_dir=synthetic_dir, sample_size=sample_size)

        return self

    def generate(self, synthetic_dir: str, sample_size: Optional[int] = None):
        """Generate and save Katabatic-compatible synthetic files."""
        synthetic_df = self.sample(num_rows=sample_size)
        x_synth, y_synth = split_synthetic(synthetic_df, self.target_col)

        if self.feature_columns is not None:
            common_cols = [col for col in self.feature_columns if col in x_synth.columns]
            if len(common_cols) == len(self.feature_columns):
                x_synth = x_synth[self.feature_columns]

        save_synthetic_outputs(x_synth, y_synth, synthetic_dir)
        return x_synth, y_synth

    def sample(self, num_rows: Optional[int] = None):
        """Return a generated synthetic dataframe without saving."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("TVAEModel must be trained before calling sample().")

        n_rows = int(num_rows or self.n_train_rows or 0)
        if n_rows <= 0:
            raise ValueError("num_rows must be positive.")

        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_rows, self.latent_dim, device=self.device)
            generated = self.model.decode(z).cpu().numpy()

        return self.preprocessor.inverse_transform(generated)

    def save(self, filepath: str):
        """Save model weights and preprocessing state."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("TVAEModel must be trained before calling save().")
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "preprocessor": self.preprocessor,
                "target_col": self.target_col,
                "feature_columns": self.feature_columns,
                "n_train_rows": self.n_train_rows,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None):
        """Load a previously saved self-contained TVAEModel."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        model = cls(
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            device=device,
        )
        model.preprocessor = checkpoint["preprocessor"]
        model.target_col = checkpoint["target_col"]
        model.feature_columns = checkpoint["feature_columns"]
        model.n_train_rows = checkpoint["n_train_rows"]
        model.model = _TabularVAE(
            input_dim=model.preprocessor.output_dim,
            hidden_dim=model.hidden_dim,
            latent_dim=model.latent_dim,
        ).to(model.device)
        model.model.load_state_dict(checkpoint["state_dict"])
        return model
