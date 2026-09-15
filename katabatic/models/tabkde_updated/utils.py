"""
Utility functions for TabKDE.

Provides helpers for:
- Array shape handling
- Distance computation (DCR)
- Data preprocessing and encoding
- Empirical copula transformation
- KDE-style sampling via GMM distances
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler

ArrayLike = np.ndarray | pd.DataFrame


# Basic helpers


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D. Reshapes 1D arrays to (1, d).

    Args:
        x: Input array.

    Returns:
        2D numpy array.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def compute_min_distances_cpu(A: ArrayLike, B: ArrayLike, k: int = 1) -> np.ndarray:
    """
    Returns distances from each point in A to its k nearest neighbour(s) in B.
    Used for computing DCR (Distance to Closest Record) distributions.

    Args:
        A: Query points, shape (n, d).
        B: Reference points, shape (m, d).
        k: Number of nearest neighbours.

    Returns:
        distances of shape (n,) if k == 1, else (n, k).
    """
    A_np = ensure_2d(np.asarray(A))
    B_np = ensure_2d(np.asarray(B))

    tree = KDTree(B_np)
    dists, _ = tree.query(A_np, k=k)
    return dists


# Preprocessing


def preprocess_data(
    df: pd.DataFrame,
    normalize: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """
    Preprocess a DataFrame for TabKDE:
    - Encodes object/category columns as integer category codes (starting from 1)
    - Optionally applies StandardScaler normalisation

    Args:
        df: Input DataFrame.
        normalize: Whether to apply StandardScaler. Default True.

    Returns:
        df_processed: Preprocessed DataFrame (all float).
        mean_: Scaler means if normalized, else None.
        var_: Scaler variances if normalized, else None.
    """
    df_copy = df.copy()

    for col in df_copy.select_dtypes(include=["object", "category"]).columns:
        df_copy[col] = df_copy[col].astype("category").cat.codes + 1

    df_copy = df_copy.astype(float)

    if not normalize:
        return df_copy, None, None

    scaler = StandardScaler()
    df_copy.loc[:, :] = scaler.fit_transform(df_copy.values)
    return df_copy, scaler.mean_, scaler.var_


# Empirical Copula Transformer


@dataclass
class EmpiricalTransformer:
    """
    Empirical CDF rank-transform per column (copula transformation).

    Maps each column to its rank in [0, 1] using the empirical CDF.
    Stores sorted column values for inverse transform.

    Usage:
        transformer = EmpiricalTransformer(df=df_numeric)
        transformer.fit()
        Z = transformer.df_ranks.values        # latent space [0,1]^d
        df_reconstructed = transformer.convert(Z_synth)
    """

    df: pd.DataFrame
    df_sorted: pd.DataFrame | None = None
    df_ranks: pd.DataFrame | None = None

    def fit(self, method: str = "average") -> pd.DataFrame:
        """
        Fit the empirical CDF transform.

        Args:
            method: Ranking method passed to DataFrame.rank(). Default 'average'.

        Returns:
            df_ranks: Rank-transformed DataFrame with values in (0, 1].
        """
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method=method).astype(int) / self.df.shape[0]
        return self.df_ranks

    @staticmethod
    def inverse_empirical(u: float, sorted_col: np.ndarray) -> float:
        """
        Inverse empirical CDF for a single column value.

        Args:
            u: Quantile value in [0, 1].
            sorted_col: Sorted original column values.

        Returns:
            Reconstructed original-space value.
        """
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n
        return float(np.interp(u, ecdf, sorted_col))

    def convert(self, u_vectors: np.ndarray) -> pd.DataFrame:
        """
        Inverse copula transform: map latent points back to original feature space.

        Args:
            u_vectors: Sampled latent points, shape (n_samples, d).

        Returns:
            DataFrame in original feature space with same columns as input df.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.df_sorted is None:
            raise RuntimeError("EmpiricalTransformer not fitted. Call fit() first.")

        u_vectors = ensure_2d(np.asarray(u_vectors))

        transformed = []
        for u_vec in u_vectors:
            row = [
                self.inverse_empirical(float(u), self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            transformed.append(row)

        return pd.DataFrame(np.asarray(transformed), columns=self.df.columns)


# KDE Sampler


def sample_points_via_dcp_distribution(
    X: np.ndarray,
    n_samples: int,
    gmm_model,
    noise_std: float = 0.01,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Core TabKDE sampler:
    - Picks random anchor points from X (latent space Z)
    - Samples radius r from the fitted GMM over DCR distances
    - Samples a random unit direction
    - Returns perturbed points: anchor + r * direction + noise

    Args:
        X: Training data in latent space Z, shape (n, d).
        n_samples: Number of synthetic points to generate.
        gmm_model: Fitted GaussianMixture model over DCR distances.
        noise_std: Scale of Gaussian noise added to samples. Default 0.01.
        random_state: Random seed. Default None.

    Returns:
        Synthetic latent points, shape (n_samples, d).
    """
    rng = np.random.default_rng(random_state)
    X = ensure_2d(np.asarray(X))
    m, d = X.shape

    # Random anchor points from training latent space
    idx = rng.integers(0, m, size=n_samples)
    anchors = X[idx]

    # Random unit directions
    directions = rng.normal(size=(n_samples, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = directions / norms

    # Radii sampled from fitted GMM over DCR distances
    distances, _ = gmm_model.sample(n_samples)
    distances = np.abs(np.asarray(distances)).reshape(n_samples, 1)

    # Small Gaussian noise for diversity
    noise = rng.normal(scale=noise_std, size=(n_samples, d))

    return anchors + distances * directions + noise
