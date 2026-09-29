"""
FairTabDiffusion
=================

Katabatic implementation of a fairness-aware tabular diffusion model, based on:

    Yang, Z., Yu, H., Guo, P., Zanna, K., Yang, X., & Sano, A. (2024).
    "Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models."
    Transactions on Machine Learning Research (TMLR).
    Paper:  https://arxiv.org/abs/2404.08254
    Code:   https://github.com/comp-well-org/fair-tab-diffusion

Overview
--------
Standard tabular diffusion models (e.g. TabDDPM) tend to inherit and amplify
demographic imbalance present in the training data. FairTabDiffusion
addresses this by:

1. Conditioning the denoising network on both the target label *and* a
   sensitive attribute (e.g. sex, race) during training.
2. Performing *balanced* sampling at generation time: the label and the
   sensitive attribute are drawn uniformly at random rather than from their
   (possibly skewed) empirical distribution, so the synthetic dataset has a
   fair joint distribution over (label, sensitive_attribute) by construction.

This implementation follows the same schema-based encode/decode approach as
``katabatic/models/ctgan`` (continuous columns quantile-normalized to
Normal, categorical columns one-hot encoded), and the same ``Model``
contract (``train``, ``evaluate``, ``sample``) defined in
``katabatic/models/base_model.py``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel

from .utils import (
    ColumnMeta,
    decode_columns,
    encode_columns,
    fit_transformers,
    infer_schema,
)


def _try_import(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _category_indices(df: pd.DataFrame, col: ColumnMeta) -> np.ndarray:
    cats = col.categories or []
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    return np.array([cat_to_idx.get(str(v), 0) for v in df[col.name]], dtype=np.int64)


class FairTabDiffusion(BaseModel):
    """Fairness-aware Gaussian diffusion model for mixed-type tabular data."""

    def __init__(
        self,
        *,
        sensitive_col: str | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        timesteps: int = 100,
        hidden: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
        balanced_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = {
            "sensitive_col": sensitive_col,
            "epochs": epochs,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "hidden": hidden,
            "lr": lr,
            "seed": seed,
            "device": device,
            "balanced_sampling": balanced_sampling,
        }

        self.schema: list[ColumnMeta] | None = None
        self._label_col: str | None = None
        self._sensitive_col: str | None = None
        self._n_classes = 1
        self._n_sensitive = 1
        self._enc_dim = 0
        self._blocks: dict[str, tuple[int, int]] = {}
        self._order: list[str] = []
        self._train_df: pd.DataFrame | None = None

        self._net = None
        self._betas = None
        self._alphas_cumprod = None
        self._device = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn"]

    def _build_schedule(self, torch):
        betas = torch.linspace(1e-4, 0.02, self.cfg["timesteps"], device=self._device)
        alphas = 1.0 - betas
        self._betas = betas
        self._alphas_cumprod = torch.cumprod(alphas, dim=0)

    def _build_net(self, torch, nn, dim: int):
        hidden = self.cfg["hidden"]
        n_classes = max(self._n_classes, 1)
        n_sensitive = max(self._n_sensitive, 1)

        class _DenoiseMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embed = nn.Sequential(
                    nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
                )
                self.label_embed = nn.Embedding(n_classes, hidden)
                self.sensitive_embed = nn.Embedding(n_sensitive, hidden)
                layers = [nn.Linear(dim + hidden, hidden), nn.SiLU()]
                for _ in range(3):
                    layers += [nn.Linear(hidden, hidden), nn.SiLU()]
                self.backbone = nn.Sequential(*layers)
                self.out = nn.Linear(hidden, dim)

            def forward(self, x, t, y, s):
                t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)
                cond = t_emb + self.label_embed(y) + self.sensitive_embed(s)
                h = torch.cat([x, cond], dim=-1)
                return self.out(self.backbone(h))

        return _DenoiseMLP().to(self._device)

    def _q_sample(self, x0, t, noise):
        sqrt_ac = self._alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_1m_ac = (1 - self._alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def train(
        self,
        data_dir: str,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> FairTabDiffusion:
        torch = _try_import("torch")
        nn = _try_import("torch.nn")
        F = _try_import("torch.nn.functional")
        if torch is None or nn is None or F is None:
            raise ImportError(
                "FairTabDiffusion requires PyTorch. Add it to your environment."
            )

        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self._device = torch.device(
            self.cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")
        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. "
                    "Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column.")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self._train_df = df.copy()
        self._label_col = df.columns[-1]

        sensitive_col = self.cfg["sensitive_col"]
        self._sensitive_col = (
            sensitive_col if sensitive_col and sensitive_col in df.columns else None
        )

        cat_cols_full = list(categorical_cols or [])
        if self._label_col not in cat_cols_full:
            cat_cols_full = [*cat_cols_full, self._label_col]

        self.schema = infer_schema(
            df, categorical_cols=cat_cols_full, continuous_cols=continuous_cols
        )
        fit_transformers(df, self.schema)

        schema_map = {c.name: c for c in self.schema}
        label_meta = schema_map[self._label_col]
        self._n_classes = max(len(label_meta.categories or []), 1)
        y_arr = _category_indices(df, label_meta)

        if self._sensitive_col is not None:
            sensitive_meta = schema_map[self._sensitive_col]
            if sensitive_meta.kind != "categorical":
                self._sensitive_col = None
                s_arr = np.zeros(len(df), dtype=np.int64)
                self._n_sensitive = 1
            else:
                self._n_sensitive = max(len(sensitive_meta.categories or []), 1)
                s_arr = _category_indices(df, sensitive_meta)
        else:
            s_arr = np.zeros(len(df), dtype=np.int64)
            self._n_sensitive = 1

        x_enc, self._blocks, self._order = encode_columns(df, self.schema)
        self._enc_dim = int(x_enc.shape[1])
        n = x_enc.shape[0]

        self._build_schedule(torch)
        self._net = self._build_net(torch, nn, self._enc_dim)
        optim = torch.optim.Adam(self._net.parameters(), lr=self.cfg["lr"])

        x_t = torch.tensor(x_enc, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y_arr, dtype=torch.long, device=self._device)
        s_t = torch.tensor(s_arr, dtype=torch.long, device=self._device)

        batch_size = self.cfg["batch_size"]
        n_batches = max(1, n // batch_size)
        for _epoch in range(self.cfg["epochs"]):
            perm = torch.randperm(n, device=self._device)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                if len(idx) == 0:
                    continue
                x0 = x_t[idx]
                y0 = y_t[idx]
                s0 = s_t[idx]

                t = torch.randint(
                    0, self.cfg["timesteps"], (x0.shape[0],), device=self._device
                )
                noise = torch.randn_like(x0)
                x_noisy = self._q_sample(x0, t, noise)
                pred_noise = self._net(x_noisy, t, y0, s0)

                loss = F.mse_loss(pred_noise, noise)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

        self.is_fitted = True

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)
            df_s = self.sample(n=len(df))
            feature_cols = [c for c in df.columns if c != self._label_col]
            x_synth = df_s[feature_cols].copy()
            y_synth = df_s[[self._label_col]].copy()
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False, header=True
            )
            meta = {
                "schema": {
                    "columns": list(df.columns),
                    "label": self._label_col,
                    "sensitive_col": self._sensitive_col,
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                },
                "training": self.cfg,
            }
            with open(
                os.path.join(synthetic_dir, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f, indent=2)
            print(f"[FairTabDiffusion] Synthetic data saved to {synthetic_dir}")

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: int | None = None,
        conditional: dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self._net is None or self.schema is None:
            raise RuntimeError("Call train() before sample().")

        torch = _try_import("torch")
        size = int(n) if n is not None else 1000
        balanced = self.cfg["balanced_sampling"]

        with torch.no_grad():
            if conditional and self._label_col in conditional:
                schema_map = {c.name: c for c in self.schema}
                label_meta = schema_map[self._label_col]
                cats = label_meta.categories or []
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                label_idx = cat_to_idx.get(str(conditional[self._label_col]), 0)
                y_gen = torch.full(
                    (size,), label_idx, dtype=torch.long, device=self._device
                )
            else:
                y_gen = torch.randint(
                    0, max(self._n_classes, 1), (size,), device=self._device
                )

            if balanced or self._sensitive_col is None or self._train_df is None:
                s_gen = torch.randint(
                    0, max(self._n_sensitive, 1), (size,), device=self._device
                )
            else:
                schema_map = {c.name: c for c in self.schema}
                sensitive_meta = schema_map[self._sensitive_col]
                cats = sensitive_meta.categories or []
                counts = (
                    self._train_df[self._sensitive_col]
                    .astype(str)
                    .value_counts()
                    .reindex(cats, fill_value=0)
                    .to_numpy()
                    + 1e-8
                )
                probs = counts / counts.sum()
                s_np = np.random.choice(len(cats), size=size, p=probs)
                s_gen = torch.tensor(s_np, dtype=torch.long, device=self._device)

            x = torch.randn(size, self._enc_dim, device=self._device)
            for t in reversed(range(self.cfg["timesteps"])):
                t_batch = torch.full((size,), t, device=self._device, dtype=torch.long)
                pred_noise = self._net(x, t_batch, y_gen, s_gen)

                alpha = 1.0 - self._betas[t]
                alpha_cumprod = self._alphas_cumprod[t]
                beta = self._betas[t]

                coef1 = 1.0 / alpha.sqrt()
                coef2 = beta / (1.0 - alpha_cumprod).sqrt()
                mean = coef1 * (x - coef2 * pred_noise)

                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + beta.sqrt() * noise
                else:
                    x = mean

            x_np = x.cpu().numpy()

        synthetic = decode_columns(x_np, self.schema, self._blocks)

        if self._label_col in synthetic.columns:
            try:
                synthetic[self._label_col] = synthetic[self._label_col].astype(int)
            except (ValueError, TypeError):
                pass

        ordered_cols = [c.name for c in self.schema]
        return synthetic[ordered_cols]
