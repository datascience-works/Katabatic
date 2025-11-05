"""
Production-level MedGAN implementation for the Katabatic framework.

Based on "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks"
by Choi et al. (2017) - https://arxiv.org/abs/1703.06490
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from pathlib import Path

from katabatic.models.base_model import Model
from katabatic.models.medgan.utils import (
    Autoencoder,
    Generator,
    Discriminator,
    sample_noise
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDGAN(Model):
    """
    MedGAN: Medical Generative Adversarial Network for tabular data synthesis.

    This model consists of three main components:
    1. Autoencoder (AE) - Compresses high-dimensional binary/count data
    2. Generator (G) - Generates synthetic data in the AE's latent space
    3. Discriminator (D) - Distinguishes real from synthetic latent representations

    The model is trained in two phases:
    1. Pre-train the autoencoder on real data
    2. Train the GAN (Generator + Discriminator) in the latent space
    """

    def __init__(
        self,
        # Architecture hyperparameters
        encoder_dim: int = 128,
        latent_dim: int = 128,
        generator_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 128,
        generator_num_layers: int = 2,
        discriminator_num_layers: int = 2,

        # Training hyperparameters
        ae_pretrain_epochs: int = 100,
        gan_epochs: int = 1000,
        batch_size: int = 1000,
        ae_lr: float = 1e-3,
        generator_lr: float = 1e-3,
        discriminator_lr: float = 1e-3,

        # Regularization
        dropout: float = 0.1,
        bn_decay: float = 0.99,

        # Other
        random_state: int = 42,
        device: Optional[str] = None
    ):
        super().__init__()

        # Hyperparameters
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.generator_num_layers = generator_num_layers
        self.discriminator_num_layers = discriminator_num_layers

        self.ae_pretrain_epochs = ae_pretrain_epochs
        self.gan_epochs = gan_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.dropout = dropout
        self.bn_decay = bn_decay
        self.random_state = random_state

        # Device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        # Models (will be initialized in train)
        self.autoencoder = None
        self.generator = None
        self.discriminator = None
        self.input_dim_ = None

    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs):
        """
        Train MedGAN model following Katabatic framework.

        Args:
            dataset_dir: Directory containing x_train.csv and y_train.csv
            synthetic_dir: Directory to save synthetic data
            **kwargs: Additional arguments
        """
        logger.info("=" * 80)
        logger.info("Training MedGAN Model")
        logger.info("=" * 80)

        # Load training data
        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        X_train = pd.read_csv(x_train_path)
        logger.info(f"Loaded training data: {X_train.shape}")

        # Combine with y_train if exists
        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)
            df_train = pd.concat([X_train, y_train], axis=1)
        else:
            df_train = X_train

        # Convert to numpy and normalize to [0, 1]
        data = df_train.values.astype(np.float32)
        self.input_dim_ = data.shape[1]

        # Store min/max for denormalization
        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)

        # Normalize to [0, 1] range (required for BCE loss)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1  # Avoid division by zero
        data_normalized = (data - self.data_min_) / data_range

        logger.info(f"Data normalized to [0, 1] range")
        logger.info(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
        logger.info(
            f"Normalized range: [{data_normalized.min():.2f}, {data_normalized.max():.2f}]")

        # Train the model
        self._fit(data_normalized)

        # Generate synthetic data
        logger.info(f"\nGenerating {len(data)} synthetic samples...")
        synth_data = self.sample(len(data))

        # Round categorical columns to integers
        # Assume all columns are categorical/discrete for tabular data
        synth_data = np.round(synth_data)

        # Save synthetic data
        os.makedirs(synthetic_dir, exist_ok=True)

        if os.path.exists(y_train_path):
            # Split back into X and y
            y_name = y_train.columns[0]
            x_synth = pd.DataFrame(synth_data[:, :-1], columns=X_train.columns)
            y_synth = pd.DataFrame(synth_data[:, -1:], columns=[y_name])

            # Ensure all training classes are present in synthetic data
            unique_train_classes = np.unique(df_train[y_name].values)
            unique_synth_classes = np.unique(y_synth[y_name].values)
            missing_classes = set(unique_train_classes) - \
                set(unique_synth_classes)

            if missing_classes:
                logger.warning(
                    f"Missing classes in synthetic data: {missing_classes}")
                logger.info(
                    "Adding dummy samples to ensure all classes are present...")

                # Add one sample for each missing class
                for cls in missing_classes:
                    # Find a training sample with this class
                    cls_idx = np.where(df_train[y_name].values == cls)[0][0]
                    dummy_row = df_train.iloc[cls_idx:cls_idx+1].values

                    # Append to synthetic data
                    dummy_x = pd.DataFrame(
                        dummy_row[:, :-1], columns=X_train.columns)
                    dummy_y = pd.DataFrame(dummy_row[:, -1:], columns=[y_name])
                    x_synth = pd.concat([x_synth, dummy_x], ignore_index=True)
                    y_synth = pd.concat([y_synth, dummy_y], ignore_index=True)

                logger.info(f"Added {len(missing_classes)} dummy samples")

            # Convert to int to ensure proper class labels
            for col in X_train.columns:
                x_synth[col] = x_synth[col].astype(int)
            y_synth[y_name] = y_synth[y_name].astype(int)

            x_synth.to_csv(os.path.join(
                synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(
                synthetic_dir, "y_synth.csv"), index=False)
        else:
            synth_df = pd.DataFrame(synth_data, columns=df_train.columns)
            # Convert all columns to int
            for col in synth_df.columns:
                synth_df[col] = synth_df[col].astype(int)
            synth_df.to_csv(os.path.join(
                synthetic_dir, "x_synth.csv"), index=False)

        logger.info(f"\nSynthetic data saved to: {synthetic_dir}")
        logger.info("Training complete!")

        return self

    def _fit(self, data: np.ndarray):
        """Internal fit method."""
        # Initialize models
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim_,
            encoder_dim=self.encoder_dim,
            latent_dim=self.latent_dim,
            bn_decay=self.bn_decay
        ).to(self.device)

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dim=self.generator_hidden_dim,
            num_layers=self.generator_num_layers,
            bn_decay=self.bn_decay
        ).to(self.device)

        self.discriminator = Discriminator(
            latent_dim=self.latent_dim,
            hidden_dim=self.discriminator_hidden_dim,
            num_layers=self.discriminator_num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Phase 1: Pretrain Autoencoder
        logger.info(
            f"\nPhase 1: Pretraining Autoencoder for {self.ae_pretrain_epochs} epochs...")
        self._pretrain_autoencoder(data)

        # Phase 2: Train GAN
        logger.info(f"\nPhase 2: Training GAN for {self.gan_epochs} epochs...")
        self._train_gan(data)

    def _pretrain_autoencoder(self, data: np.ndarray):
        """Pretrain the autoencoder."""
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.BCELoss()

        dataset = torch.tensor(data, dtype=torch.float32)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.ae_pretrain_epochs):
            self.autoencoder.train()
            total_loss = 0

            indices = torch.randperm(len(dataset))
            for i in range(n_batches):
                batch_idx = indices[i *
                                    self.batch_size:(i + 1) * self.batch_size]
                batch = dataset[batch_idx].to(self.device)

                optimizer.zero_grad()
                x_recon, _ = self.autoencoder(batch)
                loss = criterion(x_recon, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / n_batches
                logger.info(
                    f"Epoch {epoch+1}/{self.ae_pretrain_epochs}: AE Loss = {avg_loss:.6f}")

    def _train_gan(self, data: np.ndarray):
        """Train the GAN in the latent space."""
        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.generator_lr)
        optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr)
        criterion = nn.BCELoss()

        dataset = torch.tensor(data, dtype=torch.float32)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        self.autoencoder.eval()  # Freeze autoencoder

        for epoch in range(self.gan_epochs):
            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0
            g_loss_total = 0

            indices = torch.randperm(len(dataset))
            for i in range(n_batches):
                batch_idx = indices[i *
                                    self.batch_size:(i + 1) * self.batch_size]
                real_data = dataset[batch_idx].to(self.device)
                batch_len = len(real_data)

                # Get real latent representations
                with torch.no_grad():
                    real_latent = self.autoencoder.encode(real_data)

                # Train Discriminator
                optimizer_d.zero_grad()

                # Real samples
                real_labels = torch.ones(batch_len, 1, device=self.device)
                d_real = self.discriminator(real_latent)
                d_loss_real = criterion(d_real, real_labels)

                # Fake samples
                noise = sample_noise(batch_len, self.latent_dim, self.device)
                fake_latent = self.generator(noise)
                fake_labels = torch.zeros(batch_len, 1, device=self.device)
                d_fake = self.discriminator(fake_latent.detach())
                d_loss_fake = criterion(d_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()

                noise = sample_noise(batch_len, self.latent_dim, self.device)
                fake_latent = self.generator(noise)
                d_fake = self.discriminator(fake_latent)
                # Generator wants D to predict "real"
                g_loss = criterion(d_fake, real_labels)

                g_loss.backward()
                optimizer_g.step()

                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                avg_d_loss = d_loss_total / n_batches
                avg_g_loss = g_loss_total / n_batches
                logger.info(
                    f"Epoch {epoch+1}/{self.gan_epochs}: D Loss = {avg_d_loss:.6f}, G Loss = {avg_g_loss:.6f}")

    def sample(self, n: int) -> np.ndarray:
        """
        Generate synthetic samples.

        Args:
            n: Number of samples to generate

        Returns:
            Synthetic data as numpy array (denormalized to original range)
        """
        if self.autoencoder is None or self.generator is None:
            raise RuntimeError("Model must be trained before sampling")

        self.autoencoder.eval()
        self.generator.eval()

        with torch.no_grad():
            # Generate noise and pass through generator
            noise = sample_noise(n, self.latent_dim, self.device)
            fake_latent = self.generator(noise)

            # Decode latent representation to data space (normalized [0, 1])
            synthetic_data_normalized = self.autoencoder.decode(fake_latent)
            synthetic_data_normalized = synthetic_data_normalized.cpu().numpy()

        # Denormalize to original range
        data_range = self.data_max_ - self.data_min_
        synthetic_data = synthetic_data_normalized * data_range + self.data_min_

        return synthetic_data

    def evaluate(self):
        """Evaluate is handled by the pipeline's TSTREvaluation."""
        pass
