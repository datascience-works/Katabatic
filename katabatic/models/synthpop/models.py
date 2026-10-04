"""
SynthPop model integration for Katabatic.

Based on:
Nowok, B., Raab, G. M., & Dibben, C. (2016).
synthpop: Bespoke Creation of Synthetic Data in R.
Journal of Statistical Software, 74(11).
https://doi.org/10.18637/jss.v074.i11

This implementation uses the official R package synthpop
and performs sequential conditional synthesis using CART.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from katabatic.models.base_model import Model

from .utils import write_r_script

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef

logger = logging.getLogger(__name__)


class SynthPop(Model):
    """
    Katabatic wrapper for SynthPop (CART-based statistical generator).

    Generates synthetic tabular data by calling the R synthpop package
    via subprocess. Synthesis is performed using CART (Classification
    and Regression Trees) sequentially across all columns.

    Requires R (``Rscript`` on PATH) and the CRAN ``synthpop`` package —
    see the model's README for installation instructions. Unlike the other
    Katabatic models, the fitted state is not a re-runnable Python model:
    R's ``syn()`` produces one synthetic table per training run, so
    ``sample()`` resamples (with replacement when growing) from that table
    rather than generating fresh rows.
    """

    ARTIFACT_STATE_FILES = ("synthpop_state.pkl",)

    def __init__(self, seed: int = 42):
        super().__init__()
        self.check_dependencies()

        self.seed = seed
        self.target_col: str | None = None
        self.columns: list[str] | None = None

        self._synthetic_df: pd.DataFrame | None = None
        self._n_train_rows: int | None = None

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> SynthPop:
        """
        Load data from data_dir (train_full.csv, or x_train.csv + y_train.csv),
        run SynthPop's CART synthesis with R, and save the synthetic split to
        synthetic_dir.
        """
        data_dir = Path(data_dir)
        train_full = data_dir / "train_full.csv"
        x_path = data_dir / "x_train.csv"
        y_path = data_dir / "y_train.csv"

        if train_full.exists():
            df = pd.read_csv(train_full)
        else:
            if not (x_path.exists() and y_path.exists()):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. "
                    "Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError(
                    "y_train.csv must have exactly one column (the target)."
                )
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self.target_col = df.columns[-1]
        self.columns = list(df.columns)
        self._n_train_rows = len(df)

        synth_dir = Path(synthetic_dir) if synthetic_dir else None
        if synth_dir is None:
            dataset_name = data_dir.name or "dataset"
            synth_dir = Path("synthetic") / dataset_name / "synthpop"
        synth_dir.mkdir(parents=True, exist_ok=True)

        input_csv = synth_dir / "synthpop_input.csv"
        full_output_csv = synth_dir / "synthetic.csv"
        r_script_path = synth_dir / "run_synthpop.R"

        df.to_csv(input_csv, index=False)

        write_r_script(
            r_script_path=r_script_path,
            input_csv=input_csv,
            output_csv=full_output_csv,
            seed=self.seed,
        )

        logger.info("Running SynthPop (CART-based synthesis)")
        logger.info(f"Input dataset   : {input_csv}")
        logger.info(f"Synthetic output: {full_output_csv}")

        try:
            subprocess.run(["Rscript", str(r_script_path)], check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "SynthPop requires R ('Rscript' on PATH) and the CRAN 'synthpop' "
                "package. See katabatic/models/synthpop/README.md for setup."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError("SynthPop R execution failed") from e

        self._synthetic_df = pd.read_csv(full_output_csv)[self.columns]
        self.is_fitted = True

        x_synth = self._synthetic_df[self.columns[:-1]]
        y_synth = self._synthetic_df[[self.target_col]]
        x_synth.to_csv(synth_dir / "x_synth.csv", index=False)
        y_synth.to_csv(synth_dir / "y_synth.csv", index=False, header=True)

        logger.info("SynthPop synthetic data generation completed")
        self._maybe_save_artifact_state(artifact_state_dir)
        return self

    def sample(self, n_samples: int | None = None, *args, **kwargs) -> pd.DataFrame:
        """
        Return synthetic data generated by SynthPop, resampled (with
        replacement when growing) to n_samples rows. Defaults to the number
        of training rows when n_samples is omitted.
        """
        if not self.is_fitted or self._synthetic_df is None:
            raise RuntimeError("SynthPop must be trained before calling sample().")

        n_rows = int(n_samples) if n_samples is not None else self._n_train_rows
        df_synth = self._synthetic_df

        if len(df_synth) > n_rows:
            df_synth = df_synth.sample(n=n_rows, random_state=self.seed)
        elif len(df_synth) < n_rows:
            extra = n_rows - len(df_synth)
            extra_rows = df_synth.sample(
                n=extra, replace=True, random_state=self.seed + 1
            )
            df_synth = pd.concat([df_synth, extra_rows], ignore_index=True)

        return df_synth.reset_index(drop=True)

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        state = {
            "seed": self.seed,
            "target_col": self.target_col,
            "columns": self.columns,
            "n_train_rows": self._n_train_rows,
            "synthetic_df": self._synthetic_df,
        }
        os.makedirs(artifact_state_dir, exist_ok=True)
        state_path = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> SynthPop:
        state_path = cls._require_state_file(store, ref)
        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(seed=state["seed"])
        instance.target_col = state["target_col"]
        instance.columns = state["columns"]
        instance._n_train_rows = state["n_train_rows"]
        instance._synthetic_df = state["synthetic_df"]
        instance.is_fitted = True
        return instance
