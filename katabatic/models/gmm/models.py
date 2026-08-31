"""Gaussian Mixture Model synthetic data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .utils import GMMUtilsMixin


class GMMModel(GMMUtilsMixin):
    """
    Class-conditional Gaussian Mixture Model generator.

    Pipeline:
    - Detect feature types (continuous vs categorical).
    - Encode categorical features as integers.
    - Fit a Gaussian Mixture Model for each target class.
    - Generate synthetic samples using the learned class distribution.
    - Decode categorical values back to their original labels.
    """

    def __init__(
        self,
        target_col: str,
        n_components: int = 4,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        """
        Initialize the GMM synthetic data generator.

        Parameters
        ----------
        target_col : str
            Name of the target/label column.
        n_components : int
            Number of mixture components per class.
        covariance_type : str
            GMM covariance type. Supported values are
            ``full``, ``tied``, ``diag`` and ``spherical``.
        random_state : int
            Random seed used for reproducibility.
        """
        self.target_col = target_col
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

        # Filled after fit().
        self.classes_ = None
        self.class_probs_ = None
        self.features_ = None
        self.feature_types_ = {}
        self._continuous_is_int_ = {}

        # Categorical encoders.
        self._cat_value_to_int_ = {}
        self._cat_int_to_value_ = {}

        # Per-class GMMs and scalers.
        self._gmms_ = {}
        self._scalers_ = {}

        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame):
        """
        Fit class-conditional Gaussian Mixture Models.

        Parameters
        ----------
        df : pd.DataFrame
            Real dataset including the target column.
        """
        if self.target_col not in df.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not in DataFrame"
            )

        # 1. Detect feature types.
        self._detect_feature_types(df)

        # 2. Build categorical encoders.
        self._fit_categorical_encoders(df)

        # 3. Calculate target class distribution.
        class_counts = df[self.target_col].value_counts(normalize=True)
        self.classes_ = class_counts.index.to_numpy()
        self.class_probs_ = class_counts.to_numpy()

        # 4. Fit a GMM for each target class.
        self._gmms_ = {}
        self._scalers_ = {}

        for cls in self.classes_:
            df_c = df[df[self.target_col] == cls]

            # Encode features into a numeric matrix.
            X_c = self._encode_features(df_c)

            if len(X_c) < self.n_components:
                n_comp = max(1, len(X_c))
            else:
                n_comp = self.n_components

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_c.values)

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.fit(X_scaled)

            self._gmms_[cls] = gmm
            self._scalers_[cls] = scaler

    def generate(
        self,
        n_rows: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Parameters
        ----------
        n_rows : int
            Number of synthetic rows to generate.
        seed : int, optional
            Random seed used during generation.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset including the target column.
        """
        if self.classes_ is None:
            raise RuntimeError(
                "GMMModel must be fitted before calling generate()."
            )

        class_counts = self._compute_class_counts(n_rows)
        rows = []

        for index, (cls, n_c) in enumerate(
            zip(self.classes_, class_counts)
        ):
            if n_c <= 0:
                continue

            gmm = self._gmms_.get(cls)
            scaler = self._scalers_.get(cls)

            if gmm is None or scaler is None:
                continue

            # Allow Katabatic stability evaluation to control the seed.
            if seed is not None:
                gmm.random_state = seed + index

            # Sample from the GMM in scaled space.
            X_scaled = gmm.sample(n_c)[0]
            X_numeric = scaler.inverse_transform(X_scaled)

            df_numeric = pd.DataFrame(
                X_numeric,
                columns=self.features_,
            )

            # Decode categorical values and restore integer columns.
            df_decoded = self._decode_features(df_numeric)
            df_decoded[self.target_col] = cls

            rows.append(df_decoded)

        if not rows:
            raise RuntimeError(
                "No samples were generated; check fitted GMMs."
            )

        synth_df = pd.concat(rows, ignore_index=True)

        sampling_seed = (
            self.random_state if seed is None else seed
        )

        # Ensure exactly n_rows are returned.
        if len(synth_df) > n_rows:
            synth_df = synth_df.sample(
                n=n_rows,
                random_state=sampling_seed,
            ).reset_index(drop=True)

        elif len(synth_df) < n_rows:
            extra = n_rows - len(synth_df)

            extra_rows = synth_df.sample(
                n=extra,
                replace=True,
                random_state=sampling_seed + 1,
            )

            synth_df = pd.concat(
                [synth_df, extra_rows],
                ignore_index=True,
            )

        return synth_df

    def sample(
        self,
        n_rows: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the Katabatic sampling interface.

        Parameters
        ----------
        n_rows : int
            Number of synthetic rows to generate.
        seed : int, optional
            Seed used for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            Generated synthetic dataset.
        """
        return self.generate(n_rows=n_rows, seed=seed)