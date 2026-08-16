"""Implementation of a Bayesian Network synthesizer.

Learns a discrete Bayesian network structure with pgmpy's Chow-Liu tree search
(fast, robust for higher-dimensional discrete data), fits conditional
probability tables with maximum likelihood, then generates synthetic rows via
forward (ancestral) sampling. Numeric columns are quantile-discretized before
learning and decoded back to bin-midpoint values after sampling.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling

from katabatic.models.base_model import Model
from .utils import Discretizer, load_numeric_cols


class BayesianNetworkModel(Model):
    _defaults = dict(n_bins=10, seed=42)

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.cfg = {**self._defaults, **(config or {})}
        self._discretizer: Optional[Discretizer] = None
        self._bn: Optional[DiscreteBayesianNetwork] = None
        self._y_col: Optional[str] = None
        self._columns: list[str] = []

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["pgmpy"]

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "BayesianNetworkModel":
        x_path = os.path.join(dataset_dir, "x_train.csv")
        y_path = os.path.join(dataset_dir, "y_train.csv")
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        self._y_col = y.columns[0]
        full = pd.concat([X, y[self._y_col]], axis=1)

        num_cols = load_numeric_cols(dataset_dir, X)
        self._discretizer = Discretizer(num_cols, n_bins=self.cfg["n_bins"])
        discretized = self._discretizer.fit_transform(full)
        self._columns = list(discretized.columns)

        est = TreeSearch(discretized, root_node=self._columns[0])
        skeleton = est.estimate(estimator_type="chow-liu", show_progress=False)

        self._bn = DiscreteBayesianNetwork(skeleton.edges())
        self._bn.add_nodes_from(self._columns)  # keep isolated nodes if any
        self._bn.fit(
            discretized,
            estimator=DiscreteMLE(),
        )
        self.is_fitted = True

        if synthetic_dir is not None:
            n_rows = len(X)
            df_synth = self.sample(n_rows)
            os.makedirs(synthetic_dir, exist_ok=True)
            x_synth = df_synth[[c for c in self._columns if c != self._y_col]]
            y_synth = df_synth[[self._y_col]]
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return float(len(self._bn.edges()))

    def sample(self, n: int, *args, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")
        sampler = BayesianModelSampling(self._bn)
        discretized_sample = sampler.forward_sample(size=n, show_progress=False)
        discretized_sample = discretized_sample[self._columns]
        return self._discretizer.inverse_transform(discretized_sample)
