
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from katabatic.models.base_model import Model as BaseModel
from .utils import infer_schema, encode_df, decode_df, make_spans


class MaskedNet(nn.Module):
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
        harden_cats=True,
        weight_decay=1e-4,
        device="auto",
    ):
        super().__init__()
        self.task = task
        self.y_col = y_col
        # set epochs based on dataset (paper alignment)
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
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.hidden = int(hidden)
        self.ensemble_size = int(ensemble_size)
        self.n_impute_steps = int(n_impute_steps)
        self.noise_std = float(noise_std)
        self.mask_span_prob = float(mask_span_prob)
        self.balance_classes = bool(balance_classes)
        self.harden_cats = bool(harden_cats)
        self.weight_decay = float(weight_decay)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.schema = None
        self.spans = None
        self.models_by_class = {}
        self._trained = False

    def evaluate(self, *args, **kwargs):
        return {}

    def sample(self, output_dir, synthetic_dir=None, n_samples=None, *args, **kwargs):
        output_dir = Path(output_dir)
        if synthetic_dir is None:
            synthetic_dir = output_dir / "synthetic"
        synthetic_dir = Path(str(synthetic_dir).strip())
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        synth_path = synthetic_dir / "synthetic.csv"
        if synth_path.exists():
            return {
                "synthetic_csv": str(synth_path),
                "x_synth_csv": str(synthetic_dir / "x_synth.csv"),
                "y_synth_csv": str(synthetic_dir / "y_synth.csv"),
            }

        X_train = pd.read_csv(output_dir / "x_train.csv")
        y_train_df = pd.read_csv(output_dir / "y_train.csv")
        y = y_train_df.iloc[:, 0].to_numpy()

        if n_samples is None:
            n_samples = len(y)

        X_syn_df, y_syn = self._generate_conditional_df(X_train, y, n_samples=n_samples)

        x_synth_df = X_syn_df
        y_synth_df = pd.DataFrame({self.y_col: y_syn})
        synth_full = x_synth_df.copy()
        synth_full[self.y_col] = y_syn

        x_synth_df.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_synth_df.to_csv(synthetic_dir / "y_synth.csv", index=False)
        synth_full.to_csv(synthetic_dir / "synthetic.csv", index=False)

        return {
            "synthetic_csv": str(synthetic_dir / "synthetic.csv"),
            "x_synth_csv": str(synthetic_dir / "x_synth.csv"),
            "y_synth_csv": str(synthetic_dir / "y_synth.csv"),
        }

    def train(self, output_dir, *args, **kwargs):
        output_dir = Path(output_dir)
        synthetic_dir = Path(str(kwargs.get("synthetic_dir", output_dir / "synthetic")).strip())
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        X_train = pd.read_csv(output_dir / "x_train.csv")
        y_train_df = pd.read_csv(output_dir / "y_train.csv")
        y = y_train_df.iloc[:, 0].to_numpy()

        self.schema = infer_schema(X_train)
        X_enc = encode_df(X_train, self.schema)
        self.spans = make_spans(self.schema)

        X_tensor = torch.tensor(X_enc, dtype=torch.float32, device=self.device)
        classes = np.unique(y)

        self.models_by_class = {}

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

                    mask = self._sample_span_mask(batch.shape[0], d).to(self.device)
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

        X_syn_df, y_syn = self._generate_conditional_df(X_train, y, n_samples=len(y))

        x_synth_df = X_syn_df
        y_synth_df = pd.DataFrame({self.y_col: y_syn})
        synth_full = x_synth_df.copy()
        synth_full[self.y_col] = y_syn

        x_synth_df.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_synth_df.to_csv(synthetic_dir / "y_synth.csv", index=False)
        synth_full.to_csv(synthetic_dir / "synthetic.csv", index=False)

        print("MEG synthetic data saved to:", synthetic_dir)

        return {
            "synthetic_csv": str(synthetic_dir / "synthetic.csv"),
            "x_synth_csv": str(synthetic_dir / "x_synth.csv"),
            "y_synth_csv": str(synthetic_dir / "y_synth.csv"),
        }

    def _sample_span_mask(self, batch_size: int, d: int) -> torch.Tensor:
        mask = torch.zeros((batch_size, d), dtype=torch.float32)
        for (s, e) in self.spans:
            if np.random.rand() < self.mask_span_prob:
                mask[:, s:e] = 1.0
        if mask.sum() == 0:
            s, e = self.spans[np.random.randint(0, len(self.spans))]
            mask[:, s:e] = 1.0
        return mask

    def _hard_onehot_cat_blocks_torch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Force each categorical one-hot block to be valid hard one-hot (argmax).
        This helps LR/MLP a lot because features become truly discrete like real data.
        """
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
        classes, counts = np.unique(y_train, return_counts=True)

        if self.balance_classes:
            per = int(np.ceil(n_samples / len(classes)))
            target = {c: per for c in classes}
        else:
            target = {c: int(round(n_samples * (cnt / len(y_train)))) for c, cnt in zip(classes, counts)}
            total = sum(target.values())
            if total != n_samples:
                biggest = max(target, key=target.get)
                target[biggest] += (n_samples - total)

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

            # softmax to stabilize decoding, then decode by argmax
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
        n_seed = X_seed.shape[0]
        idx = torch.randint(0, n_seed, (n_samples,), device=self.device)
        X_gen = X_seed[idx].clone()
        d = X_gen.shape[1]

        # ensure seeds are already hard one-hot
        if self.harden_cats:
          X_gen = self._hard_onehot_cat_blocks_torch(X_gen)


        for _ in range(self.n_impute_steps):
            mask = self._sample_span_mask(X_gen.shape[0], d).to(self.device)
            noise = torch.randn_like(X_gen) * self.noise_std
            X_noisy = X_gen + noise * mask

            m = models[np.random.randint(0, len(models))]
            X_pred = m(X_noisy, mask)

            X_gen = X_gen * (1 - mask) + X_pred * mask

            #  critical: harden categoricals at EVERY step
            X_gen = self._hard_onehot_cat_blocks_torch(X_gen)

        return X_gen

    def _softmax_cat_blocks(self, X: np.ndarray) -> np.ndarray:
        X2 = X.copy()
        for c in self.schema.cat_cols:
            s, e = self.schema.cat_blocks[c]
            block = X2[:, s:e]
            block = block - block.max(axis=1, keepdims=True)
            expb = np.exp(block)
            prob = expb / (expb.sum(axis=1, keepdims=True) + 1e-8)
            X2[:, s:e] = prob
        return X2
