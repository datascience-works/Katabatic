from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


class KDEModel:
    """
    Class-conditional KDE-based tabular data generator.

    - For continuous features: fits a 1D KDE per (feature, class).
    - For categorical features: uses class-conditional histograms.
    - Class distribution is matched to the real data.

    Ported from the katabatic-mentorship registry (Rishi_Goyal branch,
    ``katabatic/models/kde_Rishi/kde_model.py``). The original detects
    categorical columns by dtype, which breaks on Katabatic's pipeline
    data: columns are already integer-encoded, so a categorical code and
    a real continuous value are indistinguishable by dtype alone. This
    version accepts an explicit ``categorical_cols`` set instead, sourced
    from the dataset's ``info.json`` where available (see ``models.py``).
    """

    def __init__(
        self,
        target_col: str,
        categorical_cols: set[str] | None = None,
        kernel: str = "gaussian",
        bandwidth: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.class_probs_: np.ndarray | None = None
        self.feature_types_: dict[str, str] = {}
        self.features_: list[str] = []

        # {col: {class_value: KDE}}
        self.continuous_kdes_: dict[str, dict] = {}
        # {col: {class_value: (values, probs)}}
        self.categorical_dists_: dict[str, dict] = {}
        self._continuous_is_int_: dict[str, bool] = {}

        self._rng = np.random.default_rng(random_state)

    def _detect_feature_types(self, df: pd.DataFrame) -> None:
        self.features_ = [c for c in df.columns if c != self.target_col]
        self.feature_types_ = {}
        self._continuous_is_int_ = {}

        for col in self.features_:
            if self.categorical_cols is not None:
                is_categorical = col in self.categorical_cols
            else:
                dtype = df[col].dtype
                is_categorical = dtype == "object" or str(
                    dtype).startswith("category")

            if is_categorical:
                self.feature_types_[col] = "categorical"
            else:
                self.feature_types_[col] = "continuous"
                self._continuous_is_int_[col] = np.issubdtype(
                    df[col].dtype, np.integer)

    def _choose_bandwidth(self, x: np.ndarray) -> float:
        """Rule-of-thumb bandwidth if none is provided (Scott's rule, 1D)."""
        if self.bandwidth is not None:
            return self.bandwidth

        n = len(x)
        if n <= 1:
            return 1.0

        std = np.std(x)
        if std == 0:
            std = 1.0

        bw = std * (n ** (-1.0 / 5.0))
        return max(bw, 1e-3)

    def fit(self, df: pd.DataFrame) -> KDEModel:
        if self.target_col not in df.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not in DataFrame columns.")

        self._detect_feature_types(df)

        class_counts = df[self.target_col].value_counts(normalize=True)
        self.classes_ = class_counts.index.to_numpy()
        self.class_probs_ = class_counts.to_numpy()

        self.continuous_kdes_ = {
            c: {} for c in self.features_ if self.feature_types_[c] == "continuous"
        }
        self.categorical_dists_ = {
            c: {} for c in self.features_ if self.feature_types_[c] == "categorical"
        }

        for cls in self.classes_:
            df_c = df[df[self.target_col] == cls]

            for col in self.features_:
                ftype = self.feature_types_[col]

                if ftype == "continuous":
                    x = df_c[col].to_numpy().astype(float).reshape(-1, 1)
                    if x.shape[0] == 0:
                        continue
                    bw = self._choose_bandwidth(x)
                    kde = KernelDensity(kernel=self.kernel, bandwidth=bw)
                    kde.fit(x)
                    self.continuous_kdes_[col][cls] = kde
                else:
                    vc = df_c[col].value_counts(normalize=True, dropna=False)
                    values = vc.index.to_numpy()
                    probs = vc.values.astype(float)
                    if probs.sum() == 0:
                        continue
                    probs = probs / probs.sum()
                    self.categorical_dists_[col][cls] = (values, probs)

        return self

    def _compute_class_counts(self, n_rows: int) -> np.ndarray:
        """Convert class probabilities to integer counts summing to n_rows."""
        raw_counts = self.class_probs_ * n_rows
        base_counts = np.floor(raw_counts).astype(int)
        remainder = n_rows - base_counts.sum()

        idx = 0
        while remainder > 0:
            base_counts[idx % len(base_counts)] += 1
            remainder -= 1
            idx += 1

        return base_counts

    def generate(self, n_rows: int) -> pd.DataFrame:
        if self.classes_ is None:
            raise RuntimeError(
                "KDEModel must be fitted before calling generate().")

        class_counts = self._compute_class_counts(n_rows)
        rows = []

        for cls, n_c in zip(self.classes_, class_counts):
            if n_c <= 0:
                continue

            data_c: dict[str, np.ndarray] = {}

            for col in self.features_:
                ftype = self.feature_types_[col]

                if ftype == "continuous":
                    kde = self.continuous_kdes_.get(col, {}).get(cls)
                    if kde is None:
                        data_c[col] = np.full(n_c, np.nan)
                        continue

                    samples = kde.sample(
                        n_c, random_state=self.random_state).flatten()
                    if self._continuous_is_int_.get(col, False):
                        samples = np.rint(samples).astype(int)
                    data_c[col] = samples
                else:
                    dist_dict = self.categorical_dists_.get(col, {})
                    values_probs = dist_dict.get(cls)

                    if values_probs is None:
                        all_values: list = []
                        all_probs: list = []
                        for vals, probs in dist_dict.values():
                            all_values.extend(list(vals))
                            all_probs.extend(list(probs))

                        if not all_values:
                            data_c[col] = np.array([""] * n_c)
                            continue

                        all_values = np.array(all_values)
                        all_probs = np.array(all_probs)
                        all_probs = all_probs / all_probs.sum()
                        samples = self._rng.choice(
                            all_values, size=n_c, p=all_probs)
                    else:
                        values, probs = values_probs
                        samples = self._rng.choice(values, size=n_c, p=probs)

                    data_c[col] = samples

            df_c = pd.DataFrame(data_c)
            df_c[self.target_col] = cls
            rows.append(df_c)

        synth_df = pd.concat(rows, ignore_index=True)

        if len(synth_df) > n_rows:
            synth_df = synth_df.sample(
                n=n_rows, random_state=self.random_state).reset_index(drop=True)
        elif len(synth_df) < n_rows:
            extra = n_rows - len(synth_df)
            extra_rows = synth_df.sample(
                n=extra, replace=True, random_state=self.random_state + 1)
            synth_df = pd.concat([synth_df, extra_rows], ignore_index=True)

        return synth_df
