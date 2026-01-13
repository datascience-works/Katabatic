from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def _set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalize(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    s = float(counts.sum())
    if s <= 0:
        return np.ones_like(counts, dtype=np.float64) / max(1, counts.size)
    return counts / s


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _discretize_numeric(df: pd.DataFrame, n_bins: int) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Discretise numeric columns into 0..(n_bins-1) using quantile bins.
    Returns (discretised_df, bin_meta) where bin_meta stores (min,max) per numeric col (optional).
    """
    out = df.copy()
    meta: Dict[str, Tuple[float, float]] = {}
    for c in out.columns:
        if _is_numeric_series(out[c]):
            col = out[c].astype(float)
            meta[c] = (float(col.min()), float(col.max()))
            # quantile bins; handle constant column safely
            try:
                binned = pd.qcut(col, q=n_bins, duplicates="drop")
                out[c] = binned.cat.codes.astype(int)
            except Exception:
                out[c] = 0
    return out, meta


def _mutual_information_discrete(x: np.ndarray, y: np.ndarray, x_card: int, y_card: int) -> float:
    """
    Empirical MI for discrete arrays x and y.
    """
    joint = np.zeros((x_card, y_card), dtype=np.float64)
    for i in range(x.shape[0]):
        xi = int(x[i]); yi = int(y[i])
        if 0 <= xi < x_card and 0 <= yi < y_card:
            joint[xi, yi] += 1.0
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    eps = 1e-12
    mi = 0.0
    for i in range(x_card):
        for j in range(y_card):
            pxy = joint[i, j]
            if pxy > 0:
                mi += pxy * np.log((pxy + eps) / ((px[i, 0] + eps) * (py[0, j] + eps)))
    return float(mi)


def _laplace_mechanism(rng: np.random.Generator, counts: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
    """
    Add Laplace noise to counts. Scale = sensitivity / epsilon.
    """
    if epsilon is None:
        return counts
    eps = float(epsilon)
    if eps <= 0:
        return counts
    scale = sensitivity / eps
    noisy = counts + rng.laplace(loc=0.0, scale=scale, size=counts.shape[0])
    noisy = np.clip(noisy, 0.0, None)
    return noisy

# PrivBayes core model

@dataclass
class PrivBayesModel:
    """
    Practical PrivBayes-style Bayesian Network synthesizer.

    Notes:
    - Assumes inputs are discrete integer-coded (OrdinalEncoder output) OR numeric values
      (numeric will be discretised into n_bins before learning).
    - epsilon controls DP noise in (a simple, consistent) way. If epsilon=None, runs non-DP.
    - max_parents (k) limits parent set size.
    """
    target_col: str = "target"
    epsilon: Optional[float] = 1.0
    max_parents: int = 2
    n_bins: int = 10
    seed: int = 666
    ordering: str = "greedy_mutual_information"

    # learned
    cols_: Optional[List[str]] = None
    order_: Optional[List[str]] = None
    parents_: Optional[Dict[str, List[str]]] = None
    card_: Optional[Dict[str, int]] = None
    cpts_: Optional[Dict[str, Dict[Tuple[int, ...], np.ndarray]]] = None
    has_target_: bool = False

    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        rng = _set_seed(self.seed)

        df = df.copy()
        self.has_target_ = self.target_col in df.columns

        # Discretise numeric columns (paper assumes discrete domains)
        df, _ = _discretize_numeric(df, n_bins=self.n_bins)

        # Make everything integer-coded
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        self.cols_ = list(df.columns)

        # Cardinalities
        self.card_ = {c: int(df[c].max()) + 1 for c in self.cols_}

        # Learn BN ordering + parent sets
        self.order_, self.parents_ = self._learn_structure(df, rng)

        # Estimate CPTs (with optional DP noise on counts)
        self.cpts_ = self._build_cpts(df, rng)

    def sample(self, n_samples: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.order_ is None or self.parents_ is None or self.cpts_ is None or self.card_ is None or self.cols_ is None:
            raise RuntimeError("PrivBayesModel not fitted.")

        rng = _set_seed(self.seed + 999)  # separate sampling stream
        col_index = {c: i for i, c in enumerate(self.order_)}
        out = np.zeros((n_samples, len(self.order_)), dtype=int)

        for i in range(n_samples):
            for x in self.order_:
                px = self.parents_.get(x, [])
                if not px:
                    probs = self.cpts_[x][()]
                else:
                    key = tuple(out[i, col_index[p]] for p in px)
                    probs = self.cpts_[x].get(key)
                    if probs is None:
                        probs = np.ones(self.card_[x], dtype=np.float64) / max(1, self.card_[x])
                out[i, col_index[x]] = int(rng.choice(np.arange(probs.size), p=probs))

        synth_df = pd.DataFrame(out, columns=self.order_)
        synth_df = synth_df[self.cols_]  # restore original column order

        if self.has_target_:
            y = synth_df[self.target_col].copy()
            x = synth_df.drop(columns=[self.target_col])
            return x, y
        else:
            return synth_df, None

    #
    # Structure learning

    def _learn_structure(self, df: pd.DataFrame, rng: np.random.Generator) -> Tuple[List[str], Dict[str, List[str]]]:
        cols = self.cols_
        assert cols is not None and self.card_ is not None

        remaining = cols.copy()
        # pick root randomly (paper-style)
        root = remaining.pop(int(rng.integers(0, len(remaining))))
        order = [root]

        # greedy ordering using MI with current set
        while remaining:
            best = None
            best_score = -1e18
            for c in remaining:
                score = 0.0
                xc = df[c].to_numpy()
                xc_card = self.card_[c]
                for p in order:
                    xp = df[p].to_numpy()
                    xp_card = self.card_[p]
                    score += _mutual_information_discrete(xc, xp, xc_card, xp_card)
                if score > best_score:
                    best_score = score
                    best = c
            order.append(best)
            remaining.remove(best)

        # choose parents for each node from predecessors by MI, limited to k
        parents: Dict[str, List[str]] = {order[0]: []}
        k = max(0, int(self.max_parents))

        for i in range(1, len(order)):
            child = order[i]
            prev = order[:i]
            scores = []
            x_child = df[child].to_numpy()
            x_child_card = self.card_[child]
            for p in prev:
                x_p = df[p].to_numpy()
                mi = _mutual_information_discrete(x_child, x_p, x_child_card, self.card_[p])
                scores.append((p, mi))
            scores.sort(key=lambda t: t[1], reverse=True)
            parents[child] = [p for p, _ in scores[: min(k, len(scores))]]

        return order, parents

    
    # CPT estimation (+ DP noise)
    
    def _build_cpts(self, df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Dict[Tuple[int, ...], np.ndarray]]:
        assert self.order_ is not None and self.parents_ is not None and self.card_ is not None

        d = len(self.order_)
        eps_total = self.epsilon

        # Simple consistent epsilon split across attributes (stable across datasets)
        eps_per_attr: Optional[float] = None
        if eps_total is not None:
            eps_per_attr = float(eps_total) / max(1, d)

        cpts: Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {}

        for x in self.order_:
            px = self.parents_.get(x, [])
            x_card = self.card_[x]
            cpts[x] = {}

            if not px:
                counts = df[x].value_counts().reindex(range(x_card), fill_value=0).to_numpy(dtype=np.float64)
                if eps_per_attr is not None:
                    counts = _laplace_mechanism(rng, counts, epsilon=eps_per_attr, sensitivity=1.0)
                counts = counts + 1e-6
                cpts[x][()] = _normalize(counts)
                continue

            # group by parent configuration
            group = df.groupby(px)[x].value_counts()

            # iterate unique parent configurations that appear
            parent_configs = df[px].drop_duplicates().to_numpy(dtype=int)
            for row in parent_configs:
                key = tuple(int(v) for v in row)
                counts = np.zeros(x_card, dtype=np.float64)

                # if config exists, fill counts
                try:
                    sub = group.loc[key]
                    for val, cnt in sub.items():
                        counts[int(val)] = float(cnt)
                except Exception:
                    pass

                if eps_per_attr is not None:
                    counts = _laplace_mechanism(rng, counts, epsilon=eps_per_attr, sensitivity=1.0)

                counts = counts + 1e-6
                cpts[x][key] = _normalize(counts)

        return cpts
