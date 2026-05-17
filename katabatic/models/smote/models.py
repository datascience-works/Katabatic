"""
SMOTE model implementation for the Katabatic pipeline.

SMOTE is an oversampling method that creates new minority-class samples by
interpolating between existing minority-class observations and their nearest
neighbours.

This implementation is fully self-contained — no imbalanced-learn dependency.
Only numpy, pandas, and scikit-learn (core Katabatic deps) are required.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from katabatic.models.base_model import Model as BaseModel

from .utils import (
    build_synthetic_dataframe,
    get_adjusted_k_neighbors,
    load_training_dataframe,
    resolve_synthetic_dir,
    save_metadata,
    save_synthetic_outputs,
    split_features_label,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Self-contained SMOTE engine
# ---------------------------------------------------------------------------

class _SMOTE:
    """
    Minimal SMOTE implementation using only numpy and sklearn.NearestNeighbors.

    Algorithm (Chawla et al., 2002)
    --------------------------------
    For each minority-class sample:
      1. Find its k nearest neighbours within the same class.
      2. Pick one neighbour at random.
      3. Generate a new point by linear interpolation:
             x_new = x_i + lambda * (x_neighbour - x_i)
         where lambda ~ Uniform(0, 1).

    Parameters
    ----------
    k_neighbors      : Number of nearest neighbours to consider.
    sampling_strategy: "auto" balances all classes to the majority count.
                       A float in (0, 1] sets the desired minority/majority ratio.
    random_state     : Seed for reproducibility.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str | float = "auto",
        random_state: int = 42,
    ):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit on (X, y) and return the resampled (X_res, y_res).
        Original samples are always included; synthetic rows are appended.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()

        # Determine how many synthetic samples to generate per class
        n_to_generate: Dict = {}
        for cls, count in zip(classes, counts):
            if count == majority_count:
                continue  # never oversample the majority class

            if self.sampling_strategy == "auto":
                target = majority_count
            elif isinstance(self.sampling_strategy, float):
                target = int(majority_count * self.sampling_strategy)
            else:
                target = majority_count

            needed = max(0, target - count)
            if needed > 0:
                n_to_generate[cls] = needed

        X_synth_parts = [X]
        y_synth_parts = [y]

        for cls, n_needed in n_to_generate.items():
            mask = y == cls
            X_cls = X[mask]
            n_cls = len(X_cls)

            # k must be < n_cls; get_adjusted_k_neighbors already ensures this
            # but guard here too in case _SMOTE is used standalone
            k = min(self.k_neighbors, n_cls - 1)

            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
            nn.fit(X_cls)
            # indices[:,0] is the point itself; [:,1:] are the k neighbours
            _, indices = nn.kneighbors(X_cls)
            neighbour_indices = indices[:, 1:]

            X_new = self._interpolate(X_cls, neighbour_indices, n_needed)
            y_new = np.full(n_needed, cls, dtype=y.dtype)

            X_synth_parts.append(X_new)
            y_synth_parts.append(y_new)

        return np.vstack(X_synth_parts), np.concatenate(y_synth_parts)

    def _interpolate(
        self,
        X_cls: np.ndarray,
        neighbour_indices: np.ndarray,
        n_needed: int,
    ) -> np.ndarray:
        """Generate n_needed synthetic points via random interpolation."""
        n_cls = len(X_cls)
        # Pick a random anchor for each synthetic sample
        anchor_idx = self._rng.integers(0, n_cls, size=n_needed)
        # Pick a random neighbour for each anchor
        k = neighbour_indices.shape[1]
        neighbour_col = self._rng.integers(0, k, size=n_needed)
        neighbour_idx = neighbour_indices[anchor_idx, neighbour_col]

        lam = self._rng.uniform(0, 1, size=(n_needed, 1))
        X_new = X_cls[anchor_idx] + lam * (X_cls[neighbour_idx] - X_cls[anchor_idx])
        return X_new


# ---------------------------------------------------------------------------
# Katabatic model wrapper
# ---------------------------------------------------------------------------

class SMOTEModel(BaseModel):
    """
    SMOTE: Synthetic Minority Over-sampling Technique.

    Self-contained implementation — no imbalanced-learn dependency.
    Requires only numpy, pandas, and scikit-learn (all core Katabatic deps).

    Writes to synthetic_dir:
    - x_synth.csv
    - y_synth.csv
    - metadata.json
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ) -> None:
        super().__init__()

        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self._smote: Optional[_SMOTE] = None
        self.column_names: Optional[list[str]] = None
        self.label_col: Optional[str] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn", "pandas", "numpy"]

    def train(
        self,
        output_dir: str,
        synthetic_dir: Optional[str] = None,
        label_col: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "SMOTEModel":
        """
        Fit SMOTE on the training data and save synthetic outputs.
        """
        df_train = load_training_dataframe(output_dir)
        X_train, y_train, detected_label_col = split_features_label(df_train)

        if label_col is not None:
            detected_label_col = label_col

        self.column_names = df_train.columns.tolist()
        self.label_col = detected_label_col
        self.X_train = X_train
        self.y_train = y_train

        adjusted_k = get_adjusted_k_neighbors(y_train, self.k_neighbors)

        print(f"[SMOTE] Initialising with k_neighbors={adjusted_k}...")
        self._smote = _SMOTE(
            k_neighbors=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        print(f"[SMOTE] Generating samples from {len(X_train)} training rows...")
        start_time = time.time()
        X_resampled, y_resampled = self._smote.fit_resample(X_train, y_train)
        elapsed = time.time() - start_time

        n_generated = len(X_resampled) - len(X_train)
        print(f"[SMOTE] Generated {n_generated} new rows in {elapsed:.2f} seconds.")

        # Keep the synthetic set the same size as the original training set
        rng = np.random.default_rng(self.random_state)
        if len(X_resampled) > len(X_train):
            indices = rng.choice(len(X_resampled), len(X_train), replace=False)
            X_final = X_resampled[indices]
            y_final = y_resampled[indices]
        else:
            X_final = X_resampled
            y_final = y_resampled

        df_synth = build_synthetic_dataframe(X_final, y_final, self.column_names)

        resolved_dir = resolve_synthetic_dir(output_dir, synthetic_dir)
        x_path, y_path = save_synthetic_outputs(df_synth, detected_label_col, resolved_dir)
        metadata_path = save_metadata(
            output_dir=resolved_dir,
            df_train=df_train,
            label_col=detected_label_col,
            k_neighbors=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            n_original=len(X_train),
            n_generated=n_generated,
            n_returned=len(df_synth),
        )

        self.is_fitted = True

        print("[SMOTE] Synthetic data saved:")
        print(f"  X -> {x_path}")
        print(f"  y -> {y_path}")
        print(f"  metadata -> {metadata_path}")

        return self

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X_synth, y_synth) as numpy arrays.
        """
        if not self.is_fitted or self._smote is None:
            raise RuntimeError("Call train() before sample().")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Training data was not stored correctly.")

        X_resampled, y_resampled = self._smote.fit_resample(self.X_train, self.y_train)

        # Strip original rows — keep only the synthesised ones
        n_original = len(self.X_train)
        X_synth = X_resampled[n_original:]
        y_synth = y_resampled[n_original:]

        if n is not None and n < len(X_synth):
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(X_synth), n, replace=False)
            X_synth = X_synth[indices]
            y_synth = y_synth[indices]

        return X_synth, y_synth

    def evaluate(self, X_real: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        Mean column-wise KS statistic between real and synthetic features.
        Lower is better; 0 = identical marginal distributions.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        from scipy.stats import ks_2samp

        X_synth, _ = self.sample(n=len(self.X_train))

        if X_real is None:
            X_real = self.X_train

        n_cols = min(X_real.shape[1], X_synth.shape[1])
        ks_stats = [
            ks_2samp(X_real[:, i], X_synth[:, i]).statistic
            for i in range(n_cols)
        ]
        return float(np.mean(ks_stats))
