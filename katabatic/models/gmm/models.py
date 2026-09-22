"""Gaussian Mixture Model synthetic data generator."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from katabatic.models.base_model import Model

from .utils import GMMUtilsMixin


class GMMModel(Model, GMMUtilsMixin):
    """
    Class-conditional Gaussian Mixture Model generator.

    Pipeline:
    - Detect feature types (continuous vs categorical).
    - Encode categorical features as integers.
    - Fit a Gaussian Mixture Model for each target class.
    - Generate synthetic samples using the learned class distribution.
    - Decode categorical values back to their original labels, clipping
      continuous values back into the observed training range.
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
        super().__init__()

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
        self._continuous_bounds_ = {}
        self._n_train_ = None

        # Categorical encoders.
        self._cat_value_to_int_ = {}
        self._cat_int_to_value_ = {}

        # Per-class GMMs and scalers.
        self._gmms_ = {}
        self._scalers_ = {}

        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame) -> GMMModel:
        """
        Fit class-conditional Gaussian Mixture Models directly from a
        DataFrame that already includes the target column.

        Parameters
        ----------
        df : pd.DataFrame
            Real dataset including the target column.
        """
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not in DataFrame")

        # 1. Detect feature types.
        self._detect_feature_types(df)

        # 2. Learn realistic bounds for continuous features, so generated
        #    values can be clipped back into a plausible range later.
        self._fit_continuous_bounds(df)

        # 3. Build categorical encoders.
        self._fit_categorical_encoders(df)

        # 4. Calculate target class distribution.
        class_counts = df[self.target_col].value_counts(normalize=True)
        self.classes_ = class_counts.index.to_numpy()
        self.class_probs_ = class_counts.to_numpy()

        # 5. Fit a GMM for each target class.
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

        self._n_train_ = len(df)
        self.is_fitted = True

        return self

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        target_col: str | None = None,
        **kwargs,
    ) -> GMMModel:
        """
        Train the model on the given data (Katabatic ``Model`` contract).

        Reads ``x_train.csv`` / ``y_train.csv`` from ``data_dir`` (written by
        ``benchmarks/runner.py``'s ``preprocess_and_split``), recombines them,
        and fits the class-conditional GMMs.

        Parameters
        ----------
        data_dir : str
            Directory containing ``x_train.csv`` and ``y_train.csv``.
        synthetic_dir : str, optional
            Currently unused for GMM; synthetic data is written separately
            via ``sample()`` in the benchmark scripts.
        artifact_state_dir : str, optional
            Currently unused; GMM does not yet support artifact persistence.
        target_col : str, optional
            Overrides the target column name set at construction time.
        """
        del synthetic_dir, artifact_state_dir  # not yet supported for GMM

        if target_col is not None:
            self.target_col = target_col

        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)

        combined = pd.concat([X, y], axis=1)

        return self.fit(combined)

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
            raise RuntimeError("GMMModel must be fitted before calling generate().")

        class_counts = self._compute_class_counts(n_rows)
        rows = []

        for index, (cls, n_c) in enumerate(zip(self.classes_, class_counts)):
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

            # Decode categorical values, clip continuous values back into a
            # realistic range, and restore integer columns.
            df_decoded = self._decode_features(df_numeric)
            df_decoded[self.target_col] = cls

            rows.append(df_decoded)

        if not rows:
            raise RuntimeError("No samples were generated; check fitted GMMs.")

        synth_df = pd.concat(rows, ignore_index=True)

        sampling_seed = self.random_state if seed is None else seed

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
        n_samples: int | None = None,
        *args,
        seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the Katabatic sampling interface.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic rows to generate. Defaults to the number of
            rows the model was trained on.
        seed : int, optional
            Seed used for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            Generated synthetic dataset.
        """
        if self.classes_ is None:
            raise RuntimeError("GMMModel must be fitted before calling sample().")

        if n_samples is None:
            n_samples = self._n_train_

        return self.generate(n_rows=n_samples, seed=seed)

    def evaluate(
        self,
        real_data: pd.DataFrame | None = None,
        synthetic_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> float:
        """
        Lightweight self-evaluation (Katabatic ``Model`` contract).

        Computes the mean column-wise Kolmogorov-Smirnov statistic between
        real and synthetic continuous features (lower is better; 0 means the
        distributions are indistinguishable). The full multi-dimension
        benchmark evaluation still runs separately via
        ``benchmarks/runner.py``'s ``evaluate()``.
        """
        if real_data is None:
            raise ValueError("evaluate() requires real_data for comparison.")

        if synthetic_data is None:
            synthetic_data = self.sample(n_samples=len(real_data))

        continuous_cols = [
            c for c, t in self.feature_types_.items() if t == "continuous"
        ]

        if not continuous_cols:
            return 0.0

        stats = [
            ks_2samp(
                real_data[col].dropna(),
                synthetic_data[col].dropna(),
            ).statistic
            for col in continuous_cols
            if col in synthetic_data.columns
        ]

        return float(np.mean(stats)) if stats else 0.0
