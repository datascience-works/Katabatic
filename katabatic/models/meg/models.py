"""
Production-level MEG implementation for the Katabatic framework.

Based on "MEG: Masked Ensemble Tabular Data Generator"
by Zhang et al.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from katabatic.models.base_model import Model as BaseModel
from .utils import infer_schema, encode_df, decode_df, make_spans


class MaskedNet(nn.Module):
    """
    Masked reconstruction network used inside the MEG ensemble.

    The network receives partially masked tabular features and learns
    to reconstruct only the masked feature positions.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, mask):
        x_masked = x * (1 - mask)
        out = self.net(x_masked)
        return x_masked + out * mask


class MEGModel(BaseModel):
    """
    Masked Ensemble Generator for synthetic tabular data generation.

    This implementation uses masked reconstruction with an ensemble of
    neural learners to model feature dependencies and generate synthetic
    tabular samples compatible with Katabatic's standard pipeline outputs.
    """

    def __init__(
        self,
        task="classification",
        y_col="class",
        epochs=None,
        dataset_name=None,
        batch_size=256,
        lr=2e-3,
        hidden=512,
        ensemble_size=5,
        n_impute_steps=20,
        noise_std=0.03,
        mask_span_prob=0.35,
        balance_classes=False,
        weight_decay=1e-4,
        device="auto",
        harden_cats=True,
    ):
        super().__init__()

        # Core task settings
        self.task = task
        self.y_col = y_col

        # Dataset-aligned epoch defaults
        if epochs is not None:
            self.epochs = int(epochs)
        else:
            if dataset_name is not None:
                name = dataset_name.lower()
                if name in ["adult", "shuttle"]:
                    self.epochs = 50
                else:
                    self.epochs = 100
            else:
                self.epochs = 100

        # Training and generation hyperparameters
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden = int(hidden)
        self.ensemble_size = int(ensemble_size)
        self.n_impute_steps = int(n_impute_steps)
        self.noise_std = float(noise_std)
        self.mask_span_prob = float(mask_span_prob)
        self.balance_classes = bool(balance_classes)
        self.weight_decay = float(weight_decay)
        self.harden_cats = bool(harden_cats)

        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Runtime attributes
        self.schema = None
        self.spans = None
        self.models_by_class = {}
        self._trained = False
        self.X_train_df = None
        self.y_train = None

    def evaluate(self, *args, **kwargs):
        """Evaluation is handled by Katabatic's benchmarking pipeline."""
        return {}

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate n_samples rows of synthetic data.

        Must be called after train(). Returns a pd.DataFrame with the same
        columns as the training data (features + target column).
        """
        if not self._trained:
            raise RuntimeError("MEGModel must be trained before calling sample().")

        X_syn_df, y_syn = self._generate_conditional_df(
            self.X_train_df, self.y_train, n_samples=n_samples
        )
        synth_full = X_syn_df.copy()
        synth_full[self.y_col] = y_syn
        return synth_full

    def train(self, output_dir, *args, categorical_cols: list = None, continuous_cols: list = None, **kwargs):
        """
        Train MEG using Katabatic-formatted training data.

        Args:
            output_dir: Directory containing x_train.csv and y_train.csv
        """
        output_dir = Path(output_dir)

        # Load training data
        X_train = pd.read_csv(output_dir / "x_train.csv")
        y_train_df = pd.read_csv(output_dir / "y_train.csv")
        y = y_train_df.iloc[:, 0].to_numpy()

        # Honour explicit column type hints before schema inference.
        # Without this, integer-encoded categoricals (e.g. educational-num)
        # are detected as numerical by infer_schema and returned as float,
        # which fails validate_synthetic_output's categorical dtype check.
        if categorical_cols:
            for col in categorical_cols:
                if col in X_train.columns:
                    X_train[col] = X_train[col].astype(str)

        # Cache training data (after type coercion so schema and sample() agree)
        self.X_train_df = X_train
        self.y_train = y
        self.y_col = y_train_df.columns[0]

        # Infer schema and encode mixed-type tabular data
        self.schema = infer_schema(X_train)
        X_enc = encode_df(X_train, self.schema)
        self.spans = make_spans(self.schema)

        X_tensor = torch.tensor(X_enc, dtype=torch.float32, device=self.device)
        classes = np.unique(y)

        self.models_by_class = {}

        # Train one ensemble for each class group
        for c in classes:
            idx = np.where(y == c)[0]
            Xc = X_tensor[idx]
            n, d = Xc.shape

            models = []
            opts = []

            for _ in range(self.ensemble_size):
                m = MaskedNet(d, self.hidden).to(self.device)
                models.append(m)
                opts.append(optim.AdamW(m.parameters(), lr=self.lr, weight_decay=self.weight_decay))

            for epoch in range(self.epochs):
                perm = torch.randperm(n, device=self.device)
                Xshuf = Xc[perm]

                for i in range(0, n, self.batch_size):
                    batch = Xshuf[i : i + self.batch_size]
                    if batch.shape[0] == 0:
                        continue

                    mask = self._sample_span_mask(batch.shape[0], d)
                    noise = torch.randn_like(batch) * self.noise_std
                    x_noisy = batch + noise * mask

                    for m, opt in zip(models, opts):
                        opt.zero_grad()
                        recon = m(x_noisy, mask)
                        loss = ((recon - batch) ** 2 * mask).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()

                if (epoch + 1) % 20 == 0:
                    print(f"[MEG][class={c}] Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f}")

            self.models_by_class[c] = models

        self._trained = True
        return self

    def _sample_span_mask(self, batch_size: int, d: int) -> torch.Tensor:
        """Create a binary mask over encoded feature spans."""
        mask = torch.zeros((batch_size, d), dtype=torch.float32, device=self.device)

        for (s, e) in self.spans:
            if np.random.rand() < self.mask_span_prob:
                mask[:, s:e] = 1.0

        if mask.sum() == 0:
            s, e = self.spans[np.random.randint(0, len(self.spans))]
            mask[:, s:e] = 1.0

        return mask

    def _hard_onehot_cat_blocks_torch(self, X: torch.Tensor) -> torch.Tensor:
        """Convert categorical one-hot blocks into valid hard one-hot vectors."""
        if self.schema is None:
            return X

        X2 = X.clone()

        for col in self.schema.cat_cols:
            s, e = self.schema.cat_blocks[col]
            block = X2[:, s:e]
            idx = torch.argmax(block, dim=1)

            hard = torch.zeros_like(block)
            hard[torch.arange(block.size(0), device=block.device), idx] = 1.0
            X2[:, s:e] = hard

        return X2

    def _generate_conditional_df(self, X_train_df: pd.DataFrame, y_train: np.ndarray, n_samples: int):
        """Generate a synthetic dataframe while preserving class distribution."""
        classes, counts = np.unique(y_train, return_counts=True)

        if self.balance_classes:
            per = int(np.ceil(n_samples / len(classes)))
            target = {c: per for c in classes}
            total = sum(target.values())
            if total != n_samples:
                biggest = max(target, key=target.get)
                target[biggest] += n_samples - total
        else:
            target = {c: int(round(n_samples * (cnt / len(y_train)))) for c, cnt in zip(classes, counts)}
            total = sum(target.values())

            if total != n_samples:
                biggest = max(target, key=target.get)
                target[biggest] += n_samples - total

        X_enc_all = encode_df(X_train_df, self.schema)

        X_parts, y_parts = [], []

        for c in classes:
            k = target[c]
            if k <= 0:
                continue

            idx = np.where(y_train == c)[0]
            seed = X_enc_all[idx].astype(np.float32)
            seed_t = torch.tensor(seed, dtype=torch.float32, device=self.device)

            X_gen = self._generate_from_seed(seed_t, k, self.models_by_class[c])
            X_gen = X_gen.detach().cpu().numpy()

            X_gen = self._softmax_cat_blocks(X_gen)

            X_parts.append(X_gen)
            y_parts.append(np.full((X_gen.shape[0],), c))

        X_syn = np.vstack(X_parts)
        y_syn = np.concatenate(y_parts)

        perm = np.random.permutation(len(y_syn))
        X_syn = X_syn[perm]
        y_syn = y_syn[perm]

        X_syn_df = decode_df(X_syn, self.schema)
        return X_syn_df, y_syn

    def _generate_from_seed(self, X_seed: torch.Tensor, n_samples: int, models):
        """Generate encoded synthetic samples through iterative masked imputation."""
        n_seed = X_seed.shape[0]
        idx = torch.randint(0, n_seed, (n_samples,), device=self.device)

        X_gen = X_seed[idx].clone()
        d = X_gen.shape[1]

        if self.harden_cats:
            X_gen = self._hard_onehot_cat_blocks_torch(X_gen)

        for _ in range(self.n_impute_steps):
            mask = self._sample_span_mask(X_gen.shape[0], d)
            noise = torch.randn_like(X_gen) * self.noise_std
            X_noisy = X_gen + noise * mask

            m = models[np.random.randint(0, len(models))]
            X_pred = m(X_noisy, mask)

            X_gen = X_gen * (1 - mask) + X_pred * mask

            if self.harden_cats:
                X_gen = self._hard_onehot_cat_blocks_torch(X_gen)

        return X_gen

    def _softmax_cat_blocks(self, X: np.ndarray) -> np.ndarray:
        """Apply softmax to categorical blocks before dataframe decoding."""
        X2 = X.copy()

        for c in self.schema.cat_cols:
            s, e = self.schema.cat_blocks[c]
            block = X2[:, s:e]

            block = block - block.max(axis=1, keepdims=True)
            expb = np.exp(block)
            prob = expb / (expb.sum(axis=1, keepdims=True) + 1e-8)

            X2[:, s:e] = prob

        return X2