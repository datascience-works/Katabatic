"""
TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates

Based on:
TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates
https://arxiv.org/abs/2605.17642
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .utils import (
    EmpiricalTransformer,
    compute_min_distances_cpu,
    preprocess_data,
    sample_points_via_dcp_distribution,
)


class TabKDEModel:
    """
    TabKDE: KDE-based Tabular Data Generator.

    Workflow:
        1. Preprocess full table (encode categoricals, optional normalise)
        2. Fit empirical copula transform -> latent space Z in [0,1]^d
        3. Learn DCR distribution via GMM (BIC-selected components)
        4. Sample new points via KDE with GMM-sampled radius + covariance direction
        5. Inverse copula transform back to original feature space

    No epochs. No neural training. Single-pass fit.

    Parameters
    ----------
    n_dcr_splits : int
        Number of random splits to estimate the DCR distribution. Default 10.
    max_gmm_components : int
        Max Gaussian components for GMM (BIC selection). Default 10.
    noise_std : float
        Noise scale added during KDE sampling. Default 0.01.
    random_state : int or None
        Random seed. Default None.
    """

    def __init__(
        self,
        n_dcr_splits: int = 10,
        max_gmm_components: int = 10,
        noise_std: float = 0.01,
        random_state: int | None = None,
    ):
        self.n_dcr_splits = n_dcr_splits
        self.max_gmm_components = max_gmm_components
        self.noise_std = noise_std
        self.random_state = random_state

        self._fitted = False
        self._train_df: pd.DataFrame | None = None
        self._transformer: EmpiricalTransformer | None = None
        self._Z: np.ndarray | None = None
        self._gmm: GaussianMixture | None = None
        self._col_order: list = []
        self._categorical_mappings = {}

    # Fit

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | None = None,
        **kwargs,
    ) -> None:
        """
        Fit TabKDE on the full training table.

        Parameters
        ----------
        x_train : pd.DataFrame
            Full training DataFrame (may already include label column).
        y_train : pd.Series or None
            If provided, appended as last column before fitting.
        """
        df = x_train.copy()
        if y_train is not None:
            label_col = y_train.name if y_train.name else "target"
            df[label_col] = y_train.values

        self._train_df = df.copy()
        self._col_order = list(df.columns)

        # Save original categorical values
        self._categorical_mappings = {}

        for col in df.select_dtypes(include=["object", "category"]).columns:
            categorical = df[col].astype("category")

            self._categorical_mappings[col] = {
                code + 1: value for code, value in enumerate(categorical.cat.categories)
            }

        # Step 1: Preprocess categorical columns into numeric values
        df_processed, _, _ = preprocess_data(df, normalize=False)

        # Step 2: Fit empirical copula transformer -> Z in [0,1]^d
        self._transformer = EmpiricalTransformer(df=df_processed)
        self._transformer.fit()

        # Z is the rank-transformed matrix
        Z_df = self._transformer.df_ranks
        self._Z = Z_df.values.astype(float)

        # Step 3: Learn DCR distribution via GMM
        self._gmm = self._fit_dcr_gmm(self._Z)

        self._fitted = True

    # Sample

    def sample(self, n_samples: int) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Generate n_samples synthetic rows.

        Returns
        -------
        synth_df : pd.DataFrame
            Full synthetic table in original column order.
        y_synth : None
            Labels are kept inside synth_df; adapter splits them after.
        """
        if not self._fitted:
            raise RuntimeError("TabKDEModel must be fitted before sampling.")

        # Step 4: Sample in latent space using GMM + covariance direction
        Z_synth = sample_points_via_dcp_distribution(
            X=self._Z,
            n_samples=n_samples,
            gmm_model=self._gmm,
            noise_std=self.noise_std,
            random_state=self.random_state,
        )

        # Clip to [0,1] to respect copula boundary
        Z_synth = np.clip(Z_synth, 0.0, 1.0)

        # Step 5: Inverse copula -> original feature space
        synth_df = self._transformer.convert(Z_synth)
        synth_df.columns = self._col_order

        # Restore categorical columns
        for col, mapping in self._categorical_mappings.items():
            codes = np.rint(synth_df[col]).astype(int)

            codes = np.clip(
                codes,
                min(mapping.keys()),
                max(mapping.keys()),
            )

            synth_df[col] = codes.map(mapping)

        return synth_df, None

    # DCR GMM fitting

    def _fit_dcr_gmm(self, Z: np.ndarray) -> GaussianMixture:
        """
        Estimate DCR distribution by random splits of Z,
        then fit GMM with BIC-selected number of components.
        """
        n = len(Z)
        rng = np.random.default_rng(self.random_state)
        distances = []

        for _ in range(self.n_dcr_splits):
            perm = rng.permutation(n)
            half = n // 2
            Z1 = Z[perm[:half]]
            Z2 = Z[perm[half:]]
            dists = compute_min_distances_cpu(Z2, Z1, k=1)
            distances.extend(dists.tolist())

        distances = np.array(distances).reshape(-1, 1)
        distances = distances[distances[:, 0] > 1e-10]  # remove near-zero

        best_gmm = None
        best_bic = np.inf

        for k in range(1, self.max_gmm_components + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=self.random_state,
                    max_iter=200,
                )
                gmm.fit(distances)
                bic = gmm.bic(distances)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

        return best_gmm
