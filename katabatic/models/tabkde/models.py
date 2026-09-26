"""
TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates

Based on:
TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates
https://arxiv.org/abs/2605.17642
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from katabatic.models.base_model import Model

from .utils import (
    EmpiricalTransformer,
    compute_min_distances_cpu,
    preprocess_data,
    sample_points_via_dcp_distribution,
)

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


class TabKDEModel(Model):
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

    ARTIFACT_STATE_FILES = ("tabkde_state.pkl",)

    def __init__(
        self,
        n_dcr_splits: int = 10,
        max_gmm_components: int = 10,
        noise_std: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__()
        self.n_dcr_splits = n_dcr_splits
        self.max_gmm_components = max_gmm_components
        self.noise_std = noise_std
        self.random_state = random_state
        # Persistent generator: successive sample() calls give fresh draws,
        # and a fresh model with the same random_state replays them.
        self._rng = np.random.default_rng(random_state)

        self._label_col: str | None = None
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
    ) -> TabKDEModel:
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
            self._label_col = label_col

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

        self.is_fitted = True
        return self

    # Sample

    def sample(
        self, n_samples: int | None = None, *args, seed: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic rows as a DataFrame in the original column order,
        with any label column included.

        Parameters
        ----------
        n_samples : int or None
            Rows to generate. Defaults to the number of training rows.
        seed : int or None
            Seed for this draw. Without one, the model's persistent generator
            is used, so successive calls give fresh draws. The seed reaches every
            random step in the sampler, including the GMM radius draw.
        """
        if not self.is_fitted:
            raise RuntimeError("TabKDEModel must be fitted before sampling.")
        if n_samples is None:
            n_samples = len(self._Z)

        # Step 4: Sample in latent space using GMM + covariance direction
        Z_synth = sample_points_via_dcp_distribution(
            X=self._Z,
            n_samples=n_samples,
            gmm_model=self._gmm,
            noise_std=self.noise_std,
            random_state=(
                np.random.default_rng(seed) if seed is not None else self._rng
            ),
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

        return synth_df

    # Katabatic pipeline interface

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn", "scipy"]

    def train(
        self,
        data_dir: str | Path,
        *args,
        synthetic_dir: str | Path | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> TabKDEModel:
        """
        Fit on x_train.csv / y_train.csv in ``data_dir`` and write
        x_synth.csv / y_synth.csv to ``synthetic_dir`` for TSTR.

        The label is forced through the categorical path. The inverse copula
        interpolates between training values, so a numeric label would come
        back as fractions (e.g. 0.37) that classifiers reject; the categorical
        path restores exact original labels.
        """
        data_dir = Path(data_dir)
        x_path = data_dir / "x_train.csv"
        y_path = data_dir / "y_train.csv"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Expected x_train.csv and y_train.csv in {data_dir}"
            )

        x_train = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)
        self._label_col = y_df.columns[0]
        y_train = y_df[self._label_col].astype("category")

        self.fit(x_train, y_train)

        if synthetic_dir is None:
            synthetic_dir = Path("synthetic") / (data_dir.name or "dataset") / "tabkde"
        synthetic_dir = Path(synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        synth = self.sample(len(x_train))
        synth.drop(columns=[self._label_col]).to_csv(
            synthetic_dir / "x_synth.csv", index=False
        )
        synth[[self._label_col]].to_csv(synthetic_dir / "y_synth.csv", index=False)
        synth.to_csv(synthetic_dir / "synthetic.csv", index=False)

        self._maybe_save_artifact_state(artifact_state_dir)
        return self

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted state for load_from_ref().

        The raw training frame is deliberately not saved: sampling does not
        use it. Note the artifact still necessarily contains training data,
        since TabKDE samples by perturbing training points in latent space
        (_Z) and inverts through the sorted training columns (_transformer).
        """
        state_dir = Path(artifact_state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "params": {
                "n_dcr_splits": self.n_dcr_splits,
                "max_gmm_components": self.max_gmm_components,
                "noise_std": self.noise_std,
                "random_state": self.random_state,
            },
            "label_col": self._label_col,
            "transformer": self._transformer,
            "Z": self._Z,
            "gmm": self._gmm,
            "col_order": self._col_order,
            "categorical_mappings": self._categorical_mappings,
        }
        with open(state_dir / self.ARTIFACT_STATE_FILES[0], "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> TabKDEModel:
        """Rehydrate a fitted TabKDE model from a versioned artifact."""
        state_path = cls._require_state_file(store, ref)
        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(**state["params"])
        instance._label_col = state["label_col"]
        instance._transformer = state["transformer"]
        instance._Z = state["Z"]
        instance._gmm = state["gmm"]
        instance._col_order = state["col_order"]
        instance._categorical_mappings = state["categorical_mappings"]
        instance.is_fitted = True
        return instance

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

        if best_gmm is None:
            raise ValueError(
                "TabKDE could not fit the distance-to-closest-record distribution. "
                "Check the training data isn't empty or constant, or raise n_dcr_splits."
            )
        return best_gmm
