"""Implementation of the CTAB-GAN+ model (in progress).

Status: scaffolding only. See README.md for the audit finding that led to this
file being written from scratch rather than reusing the code on the unmerged
`feature/ctabgan_plus` branch (that branch's `models.py` actually wraps SDV's
`CopulaGANSynthesizer`, not CTAB-GAN+, and pulls in `sdv` as an external
dependency outside the Katabatic framework — it fails the model-acceptance
dependency-compliance criterion).

This implementation is planned to follow the CTAB-GAN+ paper directly:
Zhao, Z., Kunar, A., Birke, R. and Chen, L. Y. (2022) "CTAB-GAN+: Enhancing
Tabular Data Synthesis". https://arxiv.org/abs/2204.00401

Planned components (tracked as the next implementation steps):
  - Mixed-type encoder: continuous columns via mode-specific normalisation
    (VGM), categorical/mixed columns via one-hot / log-transform for
    long-tailed values
  - Conditional vector sampler with training-by-sampling to correct for
    imbalanced categorical distributions
  - Generator / discriminator pair trained with WGAN-GP, plus an auxiliary
    classifier loss to keep label-column semantics consistent
  - Downstream-loss term tying discriminator training back to TSTR accuracy
"""
from __future__ import annotations

import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model


class CtabganPlus(Model):
    """CTAB-GAN+ conditional GAN for synthetic tabular data.

    Interface matches the Katabatic `Model` contract (`train`, `evaluate`,
    `sample`) and the pipeline-mode calling convention used by the other
    Katabatic models (`dataset_dir` + `synthetic_dir`), so it will drop into
    `scripts/run_ctabgan_plus.sh` and the TSTR evaluation flow without changes
    once the training loop below is filled in.
    """

    _defaults = dict(
        epochs=200,
        batch_size=256,
        seed=42,
    )

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.cfg = {**self._defaults, **(config or {})}
        self._train_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # torch is already a Katabatic core/optional dependency; no new
        # external packages required.
        return ["torch"]

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CtabganPlus":
        x_path = os.path.join(dataset_dir, "x_train.csv")
        y_path = os.path.join(dataset_dir, "y_train.csv")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(
                f"Expected x_train.csv and y_train.csv in {dataset_dir}")

        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        y_col = y.columns[0]
        self._train_df = pd.concat([X, y[y_col]], axis=1)

        raise NotImplementedError(
            "CTAB-GAN+ training loop (mixed-type encoder, conditional "
            "vector sampler, WGAN-GP generator/discriminator) is not yet "
            "implemented — see README.md 'Status' section for progress."
        )

    def evaluate(self, *args, **kwargs) -> float:
        raise NotImplementedError("Pending train() implementation.")

    def sample(
        self, n_samples: int, *args, **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        raise NotImplementedError("Pending train() implementation.")
