"""Utility methods for the GMM synthetic data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd


class GMMUtilsMixin:
    """Helper methods used by the GMM model."""

    def _detect_feature_types(self, df: pd.DataFrame):
        """Detect categorical vs continuous features, excluding target."""
        self.features_ = [c for c in df.columns if c != self.target_col]
        self.feature_types_ = {}
        self._continuous_is_int_ = {}

        for col in self.features_:
            dtype = df[col].dtype

            if dtype == "object" or str(dtype).startswith("category"):
                self.feature_types_[col] = "categorical"
            else:
                self.feature_types_[col] = "continuous"
                self._continuous_is_int_[col] = np.issubdtype(
                    dtype,
                    np.integer,
                )

    def _fit_categorical_encoders(self, df: pd.DataFrame):
        """Build categorical value-to-integer and integer-to-value mappings."""
        self._cat_value_to_int_ = {}
        self._cat_int_to_value_ = {}

        for col in self.features_:
            if self.feature_types_[col] != "categorical":
                continue

            values = pd.Series(df[col].unique()).dropna().tolist()
            values_sorted = sorted(values)

            v_to_i = {v: i for i, v in enumerate(values_sorted)}
            i_to_v = {i: v for i, v in enumerate(values_sorted)}

            self._cat_value_to_int_[col] = v_to_i
            self._cat_int_to_value_[col] = i_to_v

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features as integers."""
        X = pd.DataFrame(index=df.index)

        for col in self.features_:
            if self.feature_types_[col] == "continuous":
                X[col] = df[col].astype(float)
            else:
                v_to_i = self._cat_value_to_int_[col]
                X[col] = df[col].map(v_to_i).astype(float)

        return X

    def _decode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decode generated numeric features back to original values."""
        X_out = pd.DataFrame(index=X.index)

        for col in self.features_:
            if self.feature_types_[col] == "continuous":
                if self._continuous_is_int_.get(col, False):
                    vals = np.rint(X[col].to_numpy()).astype(int)
                    X_out[col] = vals
                else:
                    X_out[col] = X[col]
            else:
                i_to_v = self._cat_int_to_value_[col]
                vals = np.rint(X[col].to_numpy()).astype(int)

                min_idx = 0
                max_idx = max(i_to_v.keys())
                vals = np.clip(vals, min_idx, max_idx)

                decoded = [
                    i_to_v.get(i, list(i_to_v.values())[0])
                    for i in vals
                ]
                X_out[col] = decoded

        return X_out

    def _compute_class_counts(self, n_rows: int):
        """Calculate the number of rows to generate for each class."""
        raw_counts = self.class_probs_ * n_rows
        base_counts = np.floor(raw_counts).astype(int)
        remainder = n_rows - base_counts.sum()

        idx = 0

        while remainder > 0:
            base_counts[idx % len(base_counts)] += 1
            remainder -= 1
            idx += 1

        return base_counts