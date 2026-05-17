from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model
from .utils import (
    EmpiricalTransformer,
    empirical_dcr,
    inverse_copula_with_types,
    principal_guided_encoding,
    sample_kde_iterative,
)


class TabKDEModel(Model):
    """
    Katabatic implementation of TabKDE (Algorithm 12).

    Pipeline
    --------
    fit()
      1. Detect column types (numerical, ordinal, categorical).
      2. Encode categoricals via PrincipalGuidedEncoding (Alg. 5).
      3. Map E -> Z via empirical copula transform (Alg. 7).
      4. Compute sample covariance Sigma of Z.
      5. Estimate empirical DCR distribution as a GMM (Alg. 9).

    sample()
      6. Generate latent samples via boundary-aware SampleKDE-iterative (Alg. 13).
      7. Invert Z -> E via InverseECDF with probabilistic rounding (Alg. 8).
      8. Split synthetic table into X and y.

    Parameters
    ----------
    n_dcr_splits    : Random splits for DCR estimation (Alg. 9). Default 20.
    max_gmm_components : Maximum GMM components tried via BIC. Default 10.
    max_kde_attempts   : Per-sample retry limit in boundary-aware sampler. Default 50.
    ord_cols        : Column names to treat as ordinal (optional; detected
                      automatically as integer-valued columns if not supplied).
    cat_cols        : Column names to treat as categorical (optional; detected
                      automatically as object/category dtype if not supplied).
    seed            : Random seed.
    """

    def __init__(
        self,
        n_dcr_splits: int = 20,
        max_gmm_components: int = 10,
        max_kde_attempts: int = 50,
        ord_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_dcr_splits = n_dcr_splits
        self.max_gmm_components = max_gmm_components
        self.max_kde_attempts = max_kde_attempts
        self.ord_cols = ord_cols or []
        self.cat_cols = cat_cols or []
        self.seed = seed

        # Populated during fit()
        self.label_col: Optional[str] = None
        self._col_names: List[str] = []
        self._feature_col_names: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols_fit: List[str] = []
        self._ord_cols_fit: List[str] = []
        self._cat_maps: Dict[str, Dict] = {}
        self._transformer: Optional[EmpiricalTransformer] = None
        self._Z: Optional[np.ndarray] = None
        self._Sigma: Optional[np.ndarray] = None
        self._gmm = None
        self._y_train: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Katabatic pipeline entry point
    # ------------------------------------------------------------------

    def train(
        self,
        output_dir: str,
        label_col: Optional[str] = None,
        synthetic_dir: Optional[str] = None,
        **kwargs,
    ) -> "TabKDEModel":
        """
        Load x_train / y_train, fit the model, generate synthetic data,
        and optionally save to synthetic_dir.
        """
        x_train = pd.read_csv(f"{output_dir}/x_train.csv")
        y_train_df = pd.read_csv(f"{output_dir}/y_train.csv")

        if label_col is None:
            label_col = y_train_df.columns[0]

        self.label_col = label_col
        y_train = y_train_df[label_col]

        train_df = x_train.copy()
        train_df[self.label_col] = y_train.values

        self.fit(train_df)

        x_synth, y_synth = self.sample(len(x_train))

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
            pd.DataFrame(x_synth, columns=x_train.columns).to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            pd.DataFrame(y_synth, columns=[self.label_col]).to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    # ------------------------------------------------------------------
    # Core fit: implements Algorithms 5, 7, 9 from the paper
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """
        Fit the TabKDE model on a full table (features + label column).

        Steps
        -----
        1. Detect column types.
        2. PrincipalGuidedEncoding for categoricals (Alg. 5).
        3. Copula transform E -> Z (Alg. 7).
        4. Estimate covariance Sigma.
        5. Fit GMM to empirical DCR distribution (Alg. 9).
        """
        df = train_df.copy().reset_index(drop=True)
        self._col_names = list(df.columns)

        # --- Separate label from features before any encoding.
        # y is stored for empirical sampling; it never enters the KDE.
        if self.label_col and self.label_col in df.columns:
            self._y_train = df[self.label_col].copy()
            df_features = df.drop(columns=[self.label_col])
        else:
            self._y_train = None
            df_features = df

        self._feature_col_names = list(df_features.columns)

        # --- Step 1: Detect column types (features only) ---
        self._cat_cols_fit, self._ord_cols_fit, self._num_cols = (
            self._detect_column_types(df_features)
        )

        # --- Step 2: Encode ordinals as consecutive integers ---
        df_features = self._encode_ordinals(df_features)

        # --- Step 3: PrincipalGuidedEncoding for categoricals (Alg. 5) ---
        if self._cat_cols_fit:
            df_features, self._cat_maps = principal_guided_encoding(
                df_features, self._cat_cols_fit, self._num_cols
            )
        df_features = df_features.astype(float)

        # --- Step 4: Copula transform E -> Z (Alg. 7) ---
        transformer = EmpiricalTransformer(df=df_features)
        transformer.fit()
        Z_df = transformer.df_ranks
        self._transformer = transformer
        self._Z = Z_df.to_numpy(dtype=float)

        # --- Step 5: Sample covariance of Z ---
        self._Sigma = np.cov(self._Z, rowvar=False)
        # Guard: if d==1 or degenerate, use identity
        if self._Sigma.ndim < 2:
            self._Sigma = np.array([[float(self._Sigma)]])
        if not np.all(np.isfinite(self._Sigma)):
            self._Sigma = np.eye(self._Z.shape[1])

        # --- Step 6: Empirical DCR -> GMM (Alg. 9) ---
        self._gmm = empirical_dcr(
            self._Z,
            n_splits=self.n_dcr_splits,
            max_components=self.max_gmm_components,
            random_state=self.seed,
        )

        self.is_fitted = True

    # ------------------------------------------------------------------
    # Core sample: implements Algorithm 13 + inverse copula
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate n_samples synthetic rows.

        Returns
        -------
        (X_synth, y_synth) as (numpy array, numpy array).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or train() before sample().")

        # Step 1: boundary-aware latent sampling (Alg. 13)
        Z_synth = sample_kde_iterative(
            Z=self._Z,
            gmm_model=self._gmm,
            Sigma=self._Sigma,
            n_samples=n_samples,
            max_attempts=self.max_kde_attempts,
            random_state=self.seed,
        )

        # Step 2: inverse copula Z -> original feature space (Alg. 8)
        # Uses feature col names only — label was never part of the KDE.
        synth_df = inverse_copula_with_types(
            z_samples=Z_synth,
            transformer=self._transformer,
            col_names=self._feature_col_names,
            cat_cols=self._cat_cols_fit,
            ord_cols=self._ord_cols_fit,
            cat_maps=self._cat_maps,
        )

        # Step 3: decode categoricals back from float encoding to original labels
        synth_df = self._decode_categoricals(synth_df)

        # Step 4: sample y empirically from training label distribution.
        # This guarantees all classes are represented in correct proportions
        # regardless of how rare they are — the KDE cannot reliably reproduce
        # discrete class boundaries from continuous latent space.
        if self._y_train is not None:
            y_synth = (
                self._y_train
                .sample(n=n_samples, replace=True, random_state=self.seed)
                .to_numpy()
            )
        else:
            raise RuntimeError("No label data available. Call train() to load y.")

        x_synth = synth_df.to_numpy()

        return x_synth, y_synth

    # ------------------------------------------------------------------
    # evaluate: mean column-wise KS statistic (numerical cols only)
    # ------------------------------------------------------------------

    def evaluate(self, X_real: Optional[pd.DataFrame] = None, **kwargs) -> float:
        """
        Mean KS statistic across numeric feature columns.
        Lower is better; 0 = identical marginal distributions.

        Parameters
        ----------
        X_real : Real feature DataFrame. If None, uses the stored training
                 encoded data (available after fit()).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or train() before evaluate().")

        from scipy.stats import ks_2samp

        n = self._Z.shape[0]
        x_synth, _ = self.sample(n)
        x_synth_df = pd.DataFrame(x_synth)

        if X_real is None:
            # Use inverse copula of training Z as reference
            X_real_df = self._transformer.convert(self._Z)
            X_real_df.columns = [
                c for c in self._col_names if c != self.label_col
            ][:x_synth_df.shape[1]]
        else:
            X_real_df = X_real.reset_index(drop=True)

        numeric_real = X_real_df.select_dtypes(include="number")
        if numeric_real.empty:
            raise ValueError("No numeric columns found for KS evaluation.")

        ks_stats = []
        for i, col in enumerate(numeric_real.columns):
            if i < x_synth_df.shape[1]:
                stat = ks_2samp(
                    numeric_real[col].dropna().to_numpy(),
                    x_synth_df.iloc[:, i].dropna().to_numpy(),
                ).statistic
                ks_stats.append(stat)

        return float(np.mean(ks_stats)) if ks_stats else float("nan")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_column_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Infer categorical, ordinal, and numerical column names.

        Priority:
        - User-supplied cat_cols / ord_cols override detection.
        - Remaining object/category dtype cols -> categorical.
        - Remaining integer cols with low cardinality (<= 20 unique) -> ordinal.
        - Everything else -> numerical.
        """
        all_cols = list(df.columns)
        cat_cols = list(self.cat_cols) if self.cat_cols else []
        ord_cols = list(self.ord_cols) if self.ord_cols else []

        # label_col must never be remapped — always treated as numerical
        # so the original class values are preserved exactly
        assigned = set(cat_cols + ord_cols)
        if self.label_col:
            assigned.add(self.label_col)

        for col in all_cols:
            if col in assigned:
                continue
            if df[col].dtype == object or str(df[col].dtype) == "category":
                cat_cols.append(col)
                assigned.add(col)
            elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
                ord_cols.append(col)
                assigned.add(col)

        num_cols = [c for c in all_cols if c not in assigned]
        return cat_cols, ord_cols, num_cols

    def _encode_ordinals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map ordinal columns to consecutive integers 1, 2, 3, ...
        preserving their natural sort order (Algorithm 6, step 1).
        """
        df = df.copy()
        for col in self._ord_cols_fit:
            ordered = sorted(df[col].dropna().unique())
            mapping = {v: i + 1 for i, v in enumerate(ordered)}
            df[col] = df[col].map(mapping).fillna(0).astype(float)
        return df

    def _decode_categoricals(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert float-encoded categorical columns back to original string labels
        by finding the nearest encoded value in cat_maps.
        """
        df = synth_df.copy()
        for col in self._cat_cols_fit:
            if col not in self._cat_maps or col not in df.columns:
                continue
            mapping = self._cat_maps[col]
            inv_mapping = {v: k for k, v in mapping.items()}
            encoded_vals = np.array(list(inv_mapping.keys()))

            def _nearest_label(x: float) -> str:
                idx = int(np.argmin(np.abs(encoded_vals - x)))
                return inv_mapping[encoded_vals[idx]]

            df[col] = df[col].apply(_nearest_label)
        return df
