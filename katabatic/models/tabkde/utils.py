from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ArrayLike = Union[np.ndarray, pd.DataFrame]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def compute_min_distances_cpu(A: ArrayLike, B: ArrayLike, k: int = 1) -> np.ndarray:
    """
    Returns distances from each point in A to its k nearest neighbor(s) in B.
    Output shape:
      - if k == 1: (len(A),)
      - else: (len(A), k)
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    A_np = ensure_2d(A_np)
    B_np = ensure_2d(B_np)

    tree = KDTree(B_np)
    dists, _ = tree.query(A_np, k=k)
    return dists


# ---------------------------------------------------------------------------
# Encoding: T -> E
# ---------------------------------------------------------------------------

def principal_guided_encoding(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Algorithm 5 — PrincipalGuidedEncoding.

    For each categorical column, replace each category value with the mean
    projection of rows having that category onto the top PCA direction of the
    numerical features.

    Parameters
    ----------
    df       : Full encoded DataFrame (numerics already numeric).
    cat_cols : Names of categorical columns to encode.
    num_cols : Names of numerical/ordinal columns used to compute PCA direction.

    Returns
    -------
    df_encoded : DataFrame with categorical columns replaced by continuous values.
    cat_maps   : {col: {category -> float}} — needed for inverse decoding.
    """
    df_encoded = df.copy()
    cat_maps: Dict[str, Dict] = {}

    if not num_cols or not cat_cols:
        # No numerical columns to derive direction from — fall back to label encoding
        for col in cat_cols:
            cats = df[col].astype(str).unique()
            mapping = {c: float(i) for i, c in enumerate(sorted(cats))}
            df_encoded[col] = df[col].astype(str).map(mapping)
            cat_maps[col] = mapping
        return df_encoded, cat_maps

    # Top principal direction of numerical features
    X_num = df[num_cols].to_numpy(dtype=float)
    pca = PCA(n_components=1)
    pca.fit(X_num)
    u = pca.components_[0]            # shape (len(num_cols),)
    projections = X_num @ u            # shape (n_rows,)

    for col in cat_cols:
        mapping: Dict[str, float] = {}
        col_str = df[col].astype(str)
        for cat in col_str.unique():
            mask = col_str == cat
            mapping[cat] = float(projections[mask].mean()) if mask.any() else 0.0
        df_encoded[col] = col_str.map(mapping)
        cat_maps[col] = mapping

    return df_encoded, cat_maps


# ---------------------------------------------------------------------------
# Copula transform: E -> Z  and  Z -> E
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalTransformer:
    """
    Empirical CDF copula transform per column (Algorithm 7 / 8).

    fit()     : compute and store sorted values and rank matrix.
    transform : forward E -> Z  (each value mapped to its ECDF rank in [0,1]).
    convert   : inverse Z -> E  (interpolate back from sorted column values).
    """
    df: pd.DataFrame
    df_sorted: Optional[pd.DataFrame] = field(default=None, repr=False)
    df_ranks: Optional[pd.DataFrame] = field(default=None, repr=False)

    def fit(self, method: str = "average") -> pd.DataFrame:
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method=method).astype(float) / self.df.shape[0]
        return self.df_ranks

    @staticmethod
    def inverse_empirical(u: float, sorted_col: np.ndarray) -> float:
        """Scalar inverse ECDF via linear interpolation."""
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n
        return float(np.interp(u, ecdf, sorted_col))

    def convert(self, u_vectors: np.ndarray) -> pd.DataFrame:
        """Inverse transform: Z -> E  (Algorithm 8, numerical path)."""
        if self.df_sorted is None:
            raise RuntimeError("EmpiricalTransformer not fitted.")
        u_vectors = ensure_2d(np.asarray(u_vectors))

        transformed = []
        for u_vec in u_vectors:
            row = [
                self.inverse_empirical(float(u), self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            transformed.append(row)

        return pd.DataFrame(np.asarray(transformed), columns=self.df.columns)


def inverse_copula_with_types(
    z_samples: np.ndarray,
    transformer: EmpiricalTransformer,
    col_names: List[str],
    cat_cols: List[str],
    ord_cols: List[str],
    cat_maps: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Full Algorithm 8 — InverseECDF with probabilistic rounding for
    categorical and ordinal columns.

    For numerical columns  : linear interpolation between two bracketing values.
    For categorical/ordinal: probabilistic rounding to one of the two nearest
                             category values, weighted by distance.
    """
    if transformer.df_sorted is None:
        raise RuntimeError("EmpiricalTransformer not fitted.")

    z_samples = ensure_2d(np.asarray(z_samples))
    n_samples, n_cols = z_samples.shape
    result: Dict[str, list] = {c: [] for c in col_names}

    # Invert cat_maps: col -> {float_value -> original_category_str}
    inv_cat_maps: Dict[str, Dict[float, str]] = {}
    for col, mapping in cat_maps.items():
        inv_cat_maps[col] = {v: k for k, v in mapping.items()}

    for i, col in enumerate(col_names):
        sorted_col = transformer.df_sorted.iloc[:, i].values
        n = len(sorted_col)
        ecdf_vals = np.arange(1, n + 1) / n
        p_vec = z_samples[:, i]

        if col in cat_cols or col in ord_cols:
            # Sorted encoded values for this column
            sorted_enc = transformer.df_sorted.iloc[:, i].values

            for p in p_vec:
                # Clamp to [min, max]
                if p <= ecdf_vals[0]:
                    result[col].append(sorted_enc[0])
                    continue
                if p >= ecdf_vals[-1]:
                    result[col].append(sorted_enc[-1])
                    continue

                # Find bracketing indices
                idx2 = int(np.searchsorted(ecdf_vals, p, side="left"))
                idx2 = min(idx2, n - 1)
                idx1 = max(idx2 - 1, 0)

                v1, v2 = sorted_enc[idx1], sorted_enc[idx2]
                e1, e2 = ecdf_vals[idx1], ecdf_vals[idx2]

                if e2 == e1:
                    result[col].append(v1)
                    continue

                # Probability proportional to closeness (Algorithm 8)
                prob_v2 = abs(p - e1) / abs(e2 - e1)
                chosen = v2 if np.random.random() < prob_v2 else v1
                result[col].append(chosen)
        else:
            # Numerical: linear interpolation
            interp = np.interp(p_vec, ecdf_vals, sorted_col)
            result[col].extend(interp.tolist())

    return pd.DataFrame(result, columns=col_names)


# ---------------------------------------------------------------------------
# DCR distribution estimation: Algorithm 9
# ---------------------------------------------------------------------------

def empirical_dcr(
    Z: np.ndarray,
    n_splits: int = 20,
    max_components: int = 10,
    random_state: int = 42,
) -> GaussianMixture:
    """
    Algorithm 9 — EmpiricalDCR.

    Repeatedly splits Z into two halves, computes minimum distances from Z2
    to Z1, collects all distances, then fits a GMM using BIC to select k.

    Parameters
    ----------
    Z             : Latent copula-space data, shape (n, d).
    n_splits      : Number of random 50/50 splits.
    max_components: Maximum GMM components to try (BIC selection).
    random_state  : Seed for reproducibility.

    Returns
    -------
    Best-BIC GaussianMixture fitted to the empirical DCR distances.
    """
    rng = np.random.default_rng(random_state)
    Z = np.asarray(Z)
    n = len(Z)
    distances: List[float] = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = n // 2
        Z1 = Z[perm[:half]]
        Z2 = Z[perm[half:]]

        dists = compute_min_distances_cpu(Z2, Z1, k=1)
        distances.extend(dists.ravel().tolist())

    D = np.array(distances).reshape(-1, 1)
    D = D[D[:, 0] > 0]               # drop exact zeros (duplicate rows)

    best_gmm = None
    best_bic = np.inf

    for k in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                random_state=random_state,
                n_init=3,
            )
            gmm.fit(D)
            bic = gmm.bic(D)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue

    if best_gmm is None:
        # Fallback: single Gaussian
        best_gmm = GaussianMixture(n_components=1, random_state=random_state)
        best_gmm.fit(D)

    return best_gmm


# ---------------------------------------------------------------------------
# Boundary-aware KDE sampling: Algorithms 11 / 13
# ---------------------------------------------------------------------------

def _sample_direction(Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample v ~ N(0, Sigma), return u = v / ||v||."""
    v = rng.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma)
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        v = rng.standard_normal(Sigma.shape[0])
        norm_v = np.linalg.norm(v)
    return v / norm_v


def sample_kde_iterative(
    Z: np.ndarray,
    gmm_model: GaussianMixture,
    Sigma: np.ndarray,
    n_samples: int,
    max_attempts: int = 50,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Algorithm 13 — SampleKDE-iterative.

    For each sample:
      1. Uniformly pick an anchor zi from Z.
      2. Sample radius r from the GMM (absolute value to keep r > 0).
      3. Sample covariance-aware unit direction u ~ N(0, Sigma) / ||.||
      4. Propose z' = zi + r * u
      5. For out-of-[0,1] coordinates J, resample only those components of u
         (preserving the scale of the in-bounds components) and re-propose.
      6. If still out of bounds after max_attempts, restart entirely.

    Parameters
    ----------
    Z            : Latent training data, shape (n, d). Anchors are drawn from here.
    gmm_model    : Fitted GaussianMixture over DCR distances.
    Sigma        : Sample covariance of Z, shape (d, d).
    n_samples    : Number of synthetic points to generate.
    max_attempts : Per-sample retry limit before full restart.
    random_state : Optional seed.

    Returns
    -------
    Synthetic latent samples in [0, 1]^d, shape (n_samples, d).
    """
    rng = np.random.default_rng(random_state)
    Z = np.asarray(Z)
    n, d = Z.shape
    results = []

    while len(results) < n_samples:
        # Step 1: anchor
        zi = Z[rng.integers(0, n)]

        # Step 2: radius from GMM (keep positive)
        r_raw, _ = gmm_model.sample(1)
        r = float(np.abs(r_raw[0, 0]))

        # Step 3: initial direction
        u = _sample_direction(Sigma, rng)

        accepted = False
        for _ in range(max_attempts):
            z_prime = zi + r * u

            out_mask = (z_prime < 0) | (z_prime > 1)
            J = np.where(out_mask)[0]

            if len(J) == 0:
                accepted = True
                break

            # Algorithm 13, lines 7-10: resample out-of-bounds components
            w = _sample_direction(Sigma, rng)
            u_J_norm = np.linalg.norm(u[J])
            w_J_norm = np.linalg.norm(w[J])
            if w_J_norm == 0:
                continue
            s = u_J_norm / w_J_norm
            u = u.copy()
            u[J] = s * w[J]

        if accepted:
            results.append(z_prime)

    return np.array(results[:n_samples])


# ---------------------------------------------------------------------------
# Legacy helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Simple preprocessing: object -> category codes, optional StandardScaler.

    Note: TabKDE uses principal_guided_encoding + EmpiricalTransformer instead.
    This function is retained for other models in Katabatic that use it.
    """
    df_copy = df.copy()

    for col in df_copy.select_dtypes(include=["object"]).columns:
        df_copy[col] = df_copy[col].astype("category").cat.codes + 1

    df_copy = df_copy.astype(float)

    if not normalize:
        return df_copy, None, None

    scaler = StandardScaler()
    df_copy.loc[:, :] = scaler.fit_transform(df_copy.values)
    return df_copy, scaler.mean_, scaler.var_


def sample_points_via_dcp_distribution(
    X: np.ndarray,
    n_samples: int,
    gmm_model,
    noise_std: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Legacy geometry sampler (SimpleKDE style — no boundary enforcement).
    Retained for backward compatibility; TabKDE uses sample_kde_iterative.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    X = ensure_2d(X)
    m, d = X.shape

    idx = rng.integers(0, m, size=n_samples)
    anchors = X[idx]

    directions = rng.normal(size=(n_samples, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = directions / norms

    distances, _ = gmm_model.sample(n_samples)
    distances = np.asarray(distances).reshape(n_samples, 1)

    noise = rng.normal(scale=noise_std, size=(n_samples, d))

    return anchors + distances * directions + noise
