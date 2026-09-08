from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from katabatic.models.base_model import Model
from .utils import load_train_df, split_x_y, ensure_dir, try_align_columns


# ---------------------------------------------------------------------------
# Internal ARF engine (pure sklearn/numpy — no arfpy dependency)
# ---------------------------------------------------------------------------

class _ARFEngine:
    """
    Minimal Adversarial Random Forest implementation.

    Algorithm
    ---------
    1. Generate naive synthetic data by independently sampling each column
       from its empirical marginal distribution (real data).
    2. Train a Random Forest to classify real (1) vs synthetic (0).
    3. Replace synthetic data with leaf-conditional samples: for each
       synthetic point, find real rows that land in the same leaf for at
       least `leaf_thresh` fraction of trees (majority vote), then sample
       one of them per feature.
    4. Repeat 2-3 until OOB accuracy <= 0.5 + delta, or max_iters is
       reached.

    A single RNG instance is advanced across the full training loop so
    that each iteration produces genuinely different synthetic data and
    the discriminator signal can improve.
    """

    def __init__(
        self,
        num_trees: int = 30,
        max_iters: int = 10,
        delta: float = 0.0,
        min_node_size: int = 5,
        verbose: bool = True,
        seed: int = 42,
        leaf_thresh: float = 0.5,
    ):
        self.num_trees = num_trees
        self.max_iters = max_iters
        self.delta = delta
        self.min_node_size = min_node_size
        self.verbose = verbose
        self.seed = seed
        self.leaf_thresh = leaf_thresh   # fraction of trees that must agree

        self._rf: Optional[RandomForestClassifier] = None
        self._col_names: Optional[list[str]] = None
        self._col_types: Optional[dict] = None
        self._encoders: dict[int, LabelEncoder] = {}
        self._X_real_enc: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "_ARFEngine":
        """Adversarial training loop with a single persistent RNG."""
        self._col_names = list(X.columns)
        self._classify_columns(X)

        X_enc = self._encode(X)
        self._X_real_enc = X_enc.copy()

        # Single RNG advanced throughout — ensures each iteration differs
        rng = np.random.default_rng(self.seed)

        X_synth_enc = self._marginal_sample(X_enc, n=len(X_enc), rng=rng)
        prev_oob = None

        for iteration in range(self.max_iters):
            rf, oob_acc = self._train_discriminator(X_enc, X_synth_enc)
            self._rf = rf

            if self.verbose:
                print(f"[ARF] iter={iteration + 1:02d}  OOB accuracy={oob_acc:.4f}")

            # Convergence: forest barely better than chance
            if oob_acc <= 0.5 + self.delta:
                if self.verbose:
                    print(f"[ARF] Converged at iteration {iteration + 1}.")
                break

            # Early stop: no improvement from last round
            if prev_oob is not None and oob_acc >= prev_oob:
                if self.verbose:
                    print(f"[ARF] No improvement at iteration {iteration + 1}, stopping.")
                break

            prev_oob = oob_acc

            # Refine synthetic data — rng is advanced, so output differs each round
            X_synth_enc = self._leaf_sample(rf, X_enc, X_synth_enc, rng=rng)

        return self

    def forde(self) -> None:
        """No-op density estimation stub — leaves used directly in forge()."""
        if self._rf is None:
            raise RuntimeError("Call fit() before forde().")

    def forge(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate n synthetic rows via leaf-conditional sampling."""
        if self._rf is None:
            raise RuntimeError("Call fit() (and forde()) before forge().")

        # Allow an override seed for generation so repeated forge() calls
        # can produce different samples while training remains reproducible.
        rng = np.random.default_rng(seed if seed is not None else self.seed + 1)
        X_init = self._marginal_sample(self._X_real_enc, n=n, rng=rng)
        X_synth_enc = self._leaf_sample(self._rf, self._X_real_enc, X_init, rng=rng)
        return self._decode(X_synth_enc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_columns(self, X: pd.DataFrame) -> None:
        self._col_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self._col_types[col] = "numeric"
            else:
                self._col_types[col] = "categorical"

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), len(self._col_names)), dtype=float)
        for i, col in enumerate(self._col_names):
            if self._col_types[col] == "categorical":
                if i not in self._encoders:
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    self._encoders[i] = le
                out[:, i] = self._encoders[i].transform(
                    X[col].astype(str)
                ).astype(float)
            else:
                out[:, i] = X[col].to_numpy(dtype=float)
        return out

    def _decode(self, X_enc: np.ndarray) -> pd.DataFrame:
        data = {}
        for i, col in enumerate(self._col_names):
            col_data = X_enc[:, i]
            if self._col_types[col] == "categorical":
                le = self._encoders[i]
                indices = np.clip(
                    np.round(col_data).astype(int), 0, len(le.classes_) - 1
                )
                data[col] = le.inverse_transform(indices)
            else:
                data[col] = col_data
        return pd.DataFrame(data, columns=self._col_names)

    def _marginal_sample(
        self, X_enc: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample each column independently from its empirical distribution."""
        idx = rng.integers(0, len(X_enc), size=(n, X_enc.shape[1]))
        return np.stack(
            [X_enc[idx[:, j], j] for j in range(X_enc.shape[1])], axis=1
        )

    def _train_discriminator(
        self, X_real_enc: np.ndarray, X_synth_enc: np.ndarray
    ) -> tuple[RandomForestClassifier, float]:
        X_combined = np.vstack([X_real_enc, X_synth_enc])
        y_combined = np.array([1] * len(X_real_enc) + [0] * len(X_synth_enc))

        rf = RandomForestClassifier(
            n_estimators=self.num_trees,
            min_samples_leaf=self.min_node_size,
            oob_score=True,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(X_combined, y_combined)
        return rf, float(rf.oob_score_)

    def _leaf_sample(
        self,
        rf: RandomForestClassifier,
        X_real_enc: np.ndarray,
        X_synth_enc: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Majority-vote leaf matching.

        For each synthetic point, count how many trees place each real row
        in the same leaf.  Any real row agreeing on >= leaf_thresh fraction
        of trees is a candidate neighbour.  We then sample one candidate
        independently per feature.

        This is much more robust than requiring all-tree agreement, which
        fails silently on larger / higher-dimensional datasets.
        """
        real_leaves = rf.apply(X_real_enc)    # (n_real,  n_trees)
        synth_leaves = rf.apply(X_synth_enc)  # (n_synth, n_trees)
        n_trees = real_leaves.shape[1]
        threshold = int(np.ceil(self.leaf_thresh * n_trees))

        n_synth, n_features = X_synth_enc.shape
        X_new = np.empty_like(X_synth_enc)

        for s_idx in range(n_synth):
            # Count leaf agreements per real row
            agreement = np.sum(real_leaves == synth_leaves[s_idx], axis=1)
            candidate_idx = np.where(agreement >= threshold)[0]

            if len(candidate_idx) == 0:
                # Relax threshold to top-10% most similar if no match found
                top_k = max(1, int(0.1 * len(X_real_enc)))
                candidate_idx = np.argpartition(agreement, -top_k)[-top_k:]

            # Sample one candidate per feature independently for diversity
            chosen = rng.choice(candidate_idx, size=n_features, replace=True)
            X_new[s_idx] = X_real_enc[chosen, np.arange(n_features)]

        return X_new


# ---------------------------------------------------------------------------
# Katabatic ARFModel — public API unchanged
# ---------------------------------------------------------------------------

@dataclass
class ARFModel(Model):
    """
    Katabatic wrapper for Adversarial Random Forests.

    Uses a pure scikit-learn / numpy implementation — no arfpy dependency.

    Reads:
      - train_full.csv (preferred) OR x_train.csv + y_train.csv
    Writes:
      - x_synth.csv, y_synth.csv into synthetic_dir
    """
    num_trees: int = 30
    max_iters: int = 10
    delta: float = 0.0
    min_node_size: int = 5
    verbose: bool = True
    seed: int = 42
    leaf_thresh: float = 0.5

    _arf: Optional[_ARFEngine] = field(default=None, init=False, repr=False)
    _y_train: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _data_dir: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn", "numpy", "pandas"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        n_synth: Optional[int] = None,
        **kwargs,
    ) -> "ARFModel":
        self.check_dependencies()

        df = load_train_df(data_dir)
        X, y, label_col = split_x_y(df)
        y = pd.Series(y).reset_index(drop=True)

        arf = _ARFEngine(
            num_trees=self.num_trees,
            max_iters=self.max_iters,
            delta=self.delta,
            min_node_size=self.min_node_size,
            verbose=self.verbose,
            seed=self.seed,
            leaf_thresh=self.leaf_thresh,
        )
        arf.fit(X)
        arf.forde()

        self._arf = arf
        self._y_train = y
        self._data_dir = data_dir
        self.is_fitted = True

        if n_synth is None:
            n_synth = len(X)

        X_synth, y_synth = self.sample(n=n_synth)
        X_synth = try_align_columns(data_dir, X_synth)

        if synthetic_dir is not None:
            ensure_dir(synthetic_dir)
            X_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            pd.Series(y_synth, name="label").to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

            meta_path = os.path.join(synthetic_dir, "metadata.json")
            try:
                import json
                meta = {
                    "model": "arf",
                    "num_trees": self.num_trees,
                    "max_iters": self.max_iters,
                    "delta": self.delta,
                    "min_node_size": self.min_node_size,
                    "seed": self.seed,
                    "leaf_thresh": self.leaf_thresh,
                    "n_synth": int(n_synth),
                    "label_col_original": label_col,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        return self

    def sample(self, n: int = 100, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        X_synth = self._arf.forge(n=n)
        y_synth = (
            self._y_train
            .sample(n=n, replace=True, random_state=self.seed)
            .reset_index(drop=True)
        )
        return X_synth, y_synth

    def evaluate(self, X_real: Optional[pd.DataFrame] = None, **kwargs) -> float:
        """
        Mean column-wise KS statistic between real and synthetic features.
        Lower is better; 0 = identical marginal distributions.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        from scipy.stats import ks_2samp

        if X_real is None:
            if self._data_dir is None:
                raise ValueError("No data_dir stored; pass X_real explicitly.")
            df = load_train_df(self._data_dir)
            X_real, _, _ = split_x_y(df)

        X_synth, _ = self.sample(n=len(X_real))

        numeric_cols = X_real.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for KS evaluation.")

        ks_stats = [
            ks_2samp(X_real[col].dropna(), X_synth[col].dropna()).statistic
            for col in numeric_cols
            if col in X_synth.columns
        ]
        return float(np.mean(ks_stats))
