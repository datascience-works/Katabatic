"""Implementation of TabMT: masked-transformer tabular data generation.

Gulati & Roysdon (NeurIPS 2023, https://arxiv.org/abs/2312.06089). TabMT
trains a BERT-style transformer to reconstruct randomly masked columns, then
generates new rows by starting from an all-masked row and iteratively
unmasking columns in a random order, conditioning each prediction on the
columns already revealed (order-agnostic autoregressive sampling).

Simplification relative to the paper: this implementation tokenizes every
column (including numeric ones, via quantile binning) into a per-column
discrete vocabulary rather than using the paper's distribution-aware
continuous-feature embedding, and reveals one column at a time in a single
shared random order per batch rather than a fully per-row order. See
README.md "Status" for the gap to the full paper method.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from katabatic.models.base_model import Model
from .utils import ColumnTokenizer, MaskedTabularTransformer, load_column_roles


class Tabmt(Model):
    _defaults = dict(
        steps=300,
        lr=1e-3,
        batch_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_bins=12,
        min_mask_ratio=0.15,
        max_mask_ratio=0.9,
        seed=42,
    )

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.cfg = {**self._defaults, **(config or {})}
        self.device = torch.device("cpu")
        self._tokenizer: Optional[ColumnTokenizer] = None
        self._net: Optional[MaskedTabularTransformer] = None
        self._y_col: Optional[str] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch"]

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        config: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> "Tabmt":
        if config:
            self.cfg.update(config)
        torch.manual_seed(self.cfg["seed"])

        X = pd.read_csv(os.path.join(dataset_dir, "x_train.csv"))
        y = pd.read_csv(os.path.join(dataset_dir, "y_train.csv"))
        self._y_col = y.columns[0]
        full = pd.concat([X, y[self._y_col]], axis=1)

        cat_cols, num_cols = load_column_roles(dataset_dir, X)
        cat_cols = cat_cols + [self._y_col]  # model target jointly, as a column

        self._tokenizer = ColumnTokenizer(cat_cols, num_cols, n_bins=self.cfg["n_bins"])
        tokens = self._tokenizer.fit_transform(full).to(self.device)
        vocab_sizes = [self._tokenizer.vocab_sizes[c] for c in self._tokenizer.columns]
        mask_ids = torch.tensor(
            [self._tokenizer.mask_ids[c] for c in self._tokenizer.columns], device=self.device
        )

        self._net = MaskedTabularTransformer(
            vocab_sizes, d_model=self.cfg["d_model"],
            n_layers=self.cfg["n_layers"], n_heads=self.cfg["n_heads"],
        ).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg["lr"])

        n, n_cols = tokens.shape
        bsz = min(self.cfg["batch_size"], n)
        for step in range(self.cfg["steps"]):
            idx = torch.randint(0, n, (bsz,))
            batch = tokens[idx].clone()

            ratio = np.random.uniform(self.cfg["min_mask_ratio"], self.cfg["max_mask_ratio"])
            mask = torch.rand(bsz, n_cols) < ratio
            masked_batch = batch.clone()
            masked_batch[mask] = mask_ids.unsqueeze(0).expand(bsz, -1)[mask]

            logits = self._net(masked_batch)  # list of (bsz, vocab_c) per column
            loss = torch.tensor(0.0)
            n_terms = 0
            for c_idx in range(n_cols):
                col_mask = mask[:, c_idx]
                if col_mask.any():
                    loss = loss + F.cross_entropy(
                        logits[c_idx][col_mask], batch[col_mask, c_idx]
                    )
                    n_terms += 1
            loss = loss / max(n_terms, 1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % max(1, self.cfg["steps"] // 5) == 0:
                print(f"[tabmt] step {step + 1}/{self.cfg['steps']} loss={loss.item():.4f}")

        self._mask_ids = mask_ids
        self.is_fitted = True

        if synthetic_dir is not None:
            df_synth = self.sample(n)
            os.makedirs(synthetic_dir, exist_ok=True)
            feature_cols = [c for c in self._tokenizer.columns if c != self._y_col]
            df_synth[feature_cols].to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            df_synth[[self._y_col]].to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    @torch.no_grad()
    def sample(self, n: int, *args, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        n_cols = len(self._tokenizer.columns)
        tokens = self._mask_ids.unsqueeze(0).expand(n, -1).clone()

        order = torch.randperm(n_cols)
        for c_idx in order:
            logits = self._net(tokens)
            probs = F.softmax(logits[c_idx], dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            tokens[:, c_idx] = sampled

        return self._tokenizer.decode(tokens)
