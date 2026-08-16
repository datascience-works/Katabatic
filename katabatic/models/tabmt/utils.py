"""Column tokenizer and masked-transformer building blocks for TabMT.

Every column (numeric or categorical) is represented as a single discrete
token: categorical columns use their natural categories, numeric columns are
quantile-binned into `n_bins` buckets (decoded back to the bin midpoint at
sample time). This lets one shared Transformer encoder + per-column output
head operate uniformly over mixed-type rows, which is the simplification this
implementation makes relative to the TabMT paper's distribution-aware
continuous-feature embedding (see README.md "Status").
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_column_roles(dataset_dir: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    info_path = os.path.join(dataset_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cols = list(df.columns)
        cat_cols = [cols[i] for i in info.get("cat_col_idx", []) if i < len(cols)]
        num_cols = [cols[i] for i in info.get("num_col_idx", []) if i < len(cols)]
        listed = set(cat_cols) | set(num_cols)
        for c in cols:
            if c not in listed:
                (num_cols if pd.api.types.is_numeric_dtype(df[c]) else cat_cols).append(c)
        return cat_cols, num_cols
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cat_cols, num_cols


class ColumnTokenizer:
    """Tokenizes every column (numeric or categorical) into a shared integer
    vocabulary space, one vocabulary per column, with an extra MASK id.
    """

    def __init__(self, cat_cols: List[str], num_cols: List[str], n_bins: int = 12):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.columns = num_cols + cat_cols
        self.n_bins = n_bins
        self._categories: Dict[str, List[str]] = {}
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._bin_mids: Dict[str, np.ndarray] = {}
        self.vocab_sizes: Dict[str, int] = {}
        self.mask_ids: Dict[str, int] = {}

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        token_cols = []
        for c in self.num_cols:
            binned, edges = pd.qcut(df[c], q=self.n_bins, duplicates="drop", retbins=True)
            codes = binned.cat.codes.replace(-1, 0).to_numpy()
            self._bin_edges[c] = edges
            self._bin_mids[c] = (edges[:-1] + edges[1:]) / 2
            vocab = len(self._bin_mids[c])
            self.vocab_sizes[c] = vocab
            self.mask_ids[c] = vocab  # MASK token id = vocab size
            token_cols.append(codes)

        for c in self.cat_cols:
            cats = sorted(df[c].astype(str).unique().tolist())
            self._categories[c] = cats
            lookup = {v: i for i, v in enumerate(cats)}
            codes = df[c].astype(str).map(lookup).to_numpy()
            self.vocab_sizes[c] = len(cats)
            self.mask_ids[c] = len(cats)
            token_cols.append(codes)

        return torch.tensor(np.stack(token_cols, axis=1), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> pd.DataFrame:
        out = {}
        tokens = tokens.cpu().numpy()
        for i, c in enumerate(self.num_cols):
            mids = self._bin_mids[c]
            idx = np.clip(tokens[:, i], 0, len(mids) - 1)
            out[c] = mids[idx]
        offset = len(self.num_cols)
        for j, c in enumerate(self.cat_cols):
            cats = np.array(self._categories[c])
            idx = np.clip(tokens[:, offset + j], 0, len(cats) - 1)
            out[c] = cats[idx]
        return pd.DataFrame(out, columns=self.columns)


class MaskedTabularTransformer(nn.Module):
    """Shared-backbone masked transformer with a per-column embedding table
    (vocab + 1 mask token) and a per-column output head.
    """

    def __init__(self, vocab_sizes: List[int], d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_cols = len(vocab_sizes)
        self.embeds = nn.ModuleList(
            [nn.Embedding(v + 1, d_model) for v in vocab_sizes]  # +1 for MASK id
        )
        self.col_pos = nn.Embedding(self.n_cols, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, v) for v in vocab_sizes]
        )

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        # tokens: (batch, n_cols) integer ids, MASK id where masked
        pos = torch.arange(self.n_cols, device=tokens.device)
        embedded = torch.stack(
            [self.embeds[i](tokens[:, i]) for i in range(self.n_cols)], dim=1
        ) + self.col_pos(pos).unsqueeze(0)
        h = self.encoder(embedded)
        return [self.heads[i](h[:, i, :]) for i in range(self.n_cols)]
