"""
SMOTE Model Implementation for Katabatic Pipeline
Uses imbalanced-learn's SMOTE for synthetic oversampling
"""
from __future__ import annotations
from typing import Optional
import time
import warnings
import numpy as np
import pandas as pd
from katabatic.models.base_model import Model as BaseModel
from katabatic.models.smote.utils import (
    load_training_data,
    save_synthetic_data,
    save_metadata,
    resolve_synth_dir
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
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "SMOTEModel":
        """Train (fit) the SMOTE model."""

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn not found. Install with: pip install imbalanced-learn"
            )

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
        self.smote = SMOTE(
            k_neighbors=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        print(f"[SMOTE] Ready to generate samples from {len(X_train)} training samples...")
        start_time = time.time()
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Generated samples in {time.time() - start_time:.2f} seconds.")

        n_synthetic = len(X_resampled) - len(X_train)
        self.is_fitted = True

        # Subsample back to original size
        X_final, y_final = subsample_to_original_size(X_resampled, y_resampled, len(X_train))

        print(f"[SMOTE] Generated {n_synthetic} new synthetic samples...")
        print(f"[SMOTE] Returning {len(X_final)} total samples (original size with balanced classes)...")

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

        print(f"[SMOTE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples."""
        if not self.is_fitted or self.smote is None:
            raise RuntimeError("Call train() before sample().")

        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train)

        # Extract only synthetic samples
        X_synth = X_resampled[len(self.X_train):]
        y_synth = y_resampled[len(self.X_train):]

        if n is not None and n < len(X_synth):
            indices = np.random.choice(len(X_synth), n, replace=False)
            X_synth = X_synth[indices]
            y_synth = y_synth[indices]

        return pd.DataFrame(
            np.column_stack([X_synth, y_synth]), columns=self.column_names
        )
