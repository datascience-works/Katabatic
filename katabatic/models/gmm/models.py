"""Implementation of a Gaussian Mixture Model (GMM) synthesizer.

A class-conditional density-estimation baseline: fit one GaussianMixture per
target class over a label-encoded feature matrix, then sample new rows from
those fitted densities in proportion to the empirical class distribution.
Categorical columns are label-encoded, sampled in continuous space, and
rounded/clipped back to a valid category index on decode.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from katabatic.models.base_model import Model
from .utils import TabularEncoder, load_column_roles


class Gmm(Model):
    _defaults = dict(n_components=5, seed=42)

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.cfg = {**self._defaults, **(config or {})}
        self._encoder: Optional[TabularEncoder] = None
        self._class_models: dict = {}
        self._class_priors: dict = {}
        self._y_col: Optional[str] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn"]

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "Gmm":
        x_path = os.path.join(dataset_dir, "x_train.csv")
        y_path = os.path.join(dataset_dir, "y_train.csv")
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        self._y_col = y.columns[0]
        y_series = y[self._y_col]

        cat_cols, num_cols = load_column_roles(dataset_dir, X)
        self._encoder = TabularEncoder(cat_cols, num_cols)
        matrix = self._encoder.fit_transform(X)

        classes, counts = np.unique(y_series, return_counts=True)
        n_train = len(y_series)
        for cls, count in zip(classes, counts):
            mask = (y_series == cls).to_numpy()
            n_comp = min(self.cfg["n_components"], max(1, mask.sum()))
            gmm = GaussianMixture(
                n_components=n_comp,
                random_state=self.cfg["seed"],
                reg_covar=1e-3,
            )
            gmm.fit(matrix[mask])
            self._class_models[cls] = gmm
            self._class_priors[cls] = count / n_train

        self.is_fitted = True

        if synthetic_dir is not None:
            n_rows = len(X)
            df_synth = self.sample(n_rows)
            os.makedirs(synthetic_dir, exist_ok=True)
            x_synth = df_synth[self._encoder.columns]
            y_synth = df_synth[[self._y_col]] if self._y_col in df_synth else None
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            if y_synth is not None:
                y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return float(
            np.mean([m.score(m.sample(50)[0]) for m in self._class_models.values()])
        )

    def sample(self, n: int, *args, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        rng = np.random.default_rng(self.cfg["seed"])
        classes = list(self._class_models.keys())
        priors = np.array([self._class_priors[c] for c in classes])
        priors = priors / priors.sum()
        class_counts = rng.multinomial(n, priors)

        rows = []
        labels = []
        for cls, count in zip(classes, class_counts):
            if count == 0:
                continue
            sampled, _ = self._class_models[cls].sample(count)
            rows.append(sampled)
            labels.extend([cls] * count)

        matrix = np.concatenate(rows, axis=0) if rows else np.empty((0, 0))
        df = self._encoder.inverse_transform(matrix)
        df[self._y_col] = labels
        return df
