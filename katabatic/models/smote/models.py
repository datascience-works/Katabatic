"""
SMOTE Model Implementation for Katabatic Pipeline
Uses imbalanced-learn's SMOTE for synthetic oversampling
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from katabatic.models.smote.utils import (
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)

warnings.filterwarnings("ignore")


def adjust_k_neighbors(y_train: np.ndarray, k_neighbors: int) -> int:
    """
    Adjust k_neighbors to be less than the smallest class size.
    SMOTE requires k_neighbors < min_class_size.
    """
    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_size = class_counts.min()

    if min_class_size <= k_neighbors:
        adjusted_k = max(1, min_class_size - 1)
        print(f"[SMOTE] Warning: Smallest class has {min_class_size} samples.")
        print(f"[SMOTE] Adjusting k_neighbors from {k_neighbors} to {adjusted_k}")
        return adjusted_k

    return k_neighbors


def subsample_to_original_size(
    X_resampled: np.ndarray,
    y_resampled: np.ndarray,
    original_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample resampled data back down to the original dataset size.
    """
    if len(X_resampled) > original_size:
        indices = np.random.choice(len(X_resampled), original_size, replace=False)
        return X_resampled[indices], y_resampled[indices]

    return X_resampled, y_resampled


def _paper_aligned_anchor_rows(
    n_anchors: int,
    n_samples: int,
    random_state,
) -> np.ndarray:
    """Select source rows following the SMOTE paper's sampling schedule."""
    if n_anchors <= 0:
        raise ValueError("n_anchors must be greater than zero")

    complete_passes, remainder = divmod(n_samples, n_anchors)
    parts = []

    # Paper alignment:
    # For each complete 100% oversampling pass, use every minority sample once.
    if complete_passes:
        parts.append(np.tile(np.arange(n_anchors), complete_passes))

    # For a partial pass, use a random subset without replacement.
    if remainder:
        parts.append(random_state.permutation(n_anchors)[:remainder])

    if not parts:
        return np.empty(0, dtype=int)

    return np.concatenate(parts).astype(int, copy=False)


class _PaperAlignedSamplingMixin:
    """Paper-aligned source-sample scheduling for imbalanced-learn SMOTE."""

    def _make_samples(
        self,
        X,
        y_dtype,
        y_type,
        nn_data,
        nn_num,
        n_samples,
        step_size=1.0,
        y=None,
    ):
        from sklearn.utils import check_random_state

        random_state = check_random_state(self.random_state)

        rows = _paper_aligned_anchor_rows(
            nn_num.shape[0],
            n_samples,
            random_state,
        )

        # Randomly select one of the k minority-class neighbours.
        cols = random_state.randint(
            low=0,
            high=nn_num.shape[1],
            size=n_samples,
        )

        # Original SMOTE interpolation:
        # synthetic = anchor + gap * (neighbour - anchor)
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]

        X_new = self._generate_samples(
            X,
            nn_data,
            nn_num,
            rows,
            cols,
            steps,
            y_type,
            y,
        )
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new


class SMOTEModel(BaseModel):
    """
    SMOTE: Synthetic Minority Over-sampling Technique.
    Simple k-nearest neighbors interpolation for data augmentation.
    Default parameters:
        - k_neighbors: 5 (number of neighbors)
        - sampling_strategy: 'auto' (balance classes)
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

        self.smote = None
        self.column_names = None
        self.X_train = None
        self.y_train = None

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> SMOTEModel:
        """Train (fit) the SMOTE model."""

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn not found. Install with: pip install imbalanced-learn"
            )

        class PaperAlignedSMOTE(_PaperAlignedSamplingMixin, SMOTE):
            """SMOTE with source-sample scheduling aligned to the 2002 paper."""

        # Load data
        df = load_training_data(data_dir)
        self.column_names = df.columns.tolist()
        label = df.columns[-1]

        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values
        self.X_train = X_train
        self.y_train = y_train

        adjusted_k = adjust_k_neighbors(y_train, self.k_neighbors)

        # Initialise and fit SMOTE
        print(f"[SMOTE] Initializing with k_neighbors={adjusted_k}...")
        self.smote = PaperAlignedSMOTE(
            k_neighbors=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        print(
            f"[SMOTE] Ready to generate samples from {len(X_train)} training samples..."
        )
        start_time = time.time()
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Generated samples in {time.time() - start_time:.2f} seconds.")

        n_synthetic = len(X_resampled) - len(X_train)
        self.is_fitted = True

        # Subsample back to original size
        X_final, y_final = subsample_to_original_size(
            X_resampled, y_resampled, len(X_train)
        )

        print(f"[SMOTE] Generated {n_synthetic} new synthetic samples...")
        print(
            f"[SMOTE] Returning {len(X_final)} total samples (original size with balanced classes)..."
        )

        # Save outputs
        synth_dir = resolve_synth_dir(synthetic_dir, data_dir, "smote")

        x_path_out, y_path_out = save_synthetic_data(
            X_final, y_final, self.column_names, label, synth_dir
        )

        save_metadata(
            synth_dir=synth_dir,
            df=df,
            label=label,
            adjusted_k=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            n_original=len(X_train),
            n_synthetic=n_synthetic,
            n_returned=len(X_final),
        )

        print(
            f"[SMOTE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
        )
        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples."""
        if not self.is_fitted or self.smote is None:
            raise RuntimeError("Call train() before sample().")

        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train)

        X_synth = X_resampled
        y_synth = y_resampled

        if n is not None and n < len(X_synth):
            indices = np.random.choice(len(X_synth), n, replace=False)
            X_synth = X_synth[indices]
            y_synth = y_synth[indices]

        return pd.DataFrame(
            np.column_stack([X_synth, y_synth]), columns=self.column_names
        )
