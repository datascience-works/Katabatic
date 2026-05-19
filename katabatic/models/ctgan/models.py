import os

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from katabatic.models.base_model import Model


# =====================
# Generator Network
# =====================
class Generator(nn.Module):
    def __init__(self, noise_dim, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, data_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================
# Discriminator Network
# =====================
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =====================
# CTGAN Model
# =====================
class CTGANModel(Model):
    def __init__(self, epochs=50, batch_size=128, lr=1e-3, noise_dim=32, seed=42):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.noise_dim = noise_dim
        self.seed = seed

        self.generator = None
        self.discriminator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._col_names = []
        self._cat_cols = []
        self._cat_encoders = {}

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _encode(self, df):
        df = df.copy().fillna(0)
        self._cat_encoders = {}
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._cat_encoders[col] = le
        return df.astype(float)

    def train(self, split_dir, *, categorical_cols=None, continuous_cols=None, **kwargs):
        self._set_seed()
        self._cat_cols = list(categorical_cols or [])

        # Prefer the capped sample when max_train_rows was applied; fall back to full split.
        sample_path = os.path.join(split_dir, "train_sample.csv")
        full_path   = os.path.join(split_dir, "train_full.csv")
        if os.path.exists(sample_path):
            train_df = pd.read_csv(sample_path)
        elif os.path.exists(full_path):
            train_df = pd.read_csv(full_path)
        else:
            raise FileNotFoundError(f"No training CSV found in {split_dir}")

        self._col_names = list(train_df.columns)
        X_enc = self._encode(train_df)
        X_tensor = torch.tensor(X_enc.values, dtype=torch.float32).to(self.device)
        data_dim = X_tensor.shape[1]

        self.generator     = Generator(self.noise_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim).to(self.device)

        g_opt   = optim.Adam(self.generator.parameters(),     lr=self.lr)
        d_opt   = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        for epoch in range(self.epochs):
            perm     = torch.randperm(len(X_tensor))
            X_tensor = X_tensor[perm]

            for i in range(0, len(X_tensor), self.batch_size):
                real = X_tensor[i:i + self.batch_size]
                bs = real.size(0)
                real_labels = torch.ones(bs, 1).to(self.device) * 0.9
                fake_labels = torch.zeros(bs, 1).to(self.device)

                # Train Discriminator
                noise  = torch.randn(bs, self.noise_dim).to(self.device)
                fake   = self.generator(noise)
                d_loss = (loss_fn(self.discriminator(real),          real_labels) +
                          loss_fn(self.discriminator(fake.detach()), fake_labels))
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                # Train Generator
                noise  = torch.randn(bs, self.noise_dim).to(self.device)
                fake   = self.generator(noise)
                g_loss = loss_fn(self.discriminator(fake), real_labels)
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}]  "
                      f"D-loss: {d_loss.item():.4f}  G-loss: {g_loss.item():.4f}")

        self.is_fitted = True
        return self

    def sample(self, n_samples, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        self.generator.eval()
        with torch.no_grad():
            noise     = torch.randn(n_samples, self.noise_dim).to(self.device)
            synthetic = self.generator(noise).cpu().numpy()

        df = pd.DataFrame(synthetic, columns=self._col_names)

        # Inverse-transform string-encoded columns back to original values.
        for col, le in self._cat_encoders.items():
            if col in df.columns:
                codes   = df[col].round().astype(int).clip(0, len(le.classes_) - 1)
                df[col] = le.inverse_transform(codes.to_numpy())

        # Numeric categorical columns were never string-encoded; round float → int.
        for col in self._cat_cols:
            if col in df.columns and col not in self._cat_encoders:
                df[col] = df[col].round().astype(int)

        return df
