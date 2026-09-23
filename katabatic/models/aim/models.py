"""Adaptive Iterative Mechanism (AIM) model implementation for Katabatic."""

from __future__ import annotations

import itertools
import os
import pickle
from typing import Any

import pandas as pd

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.aim.utils import (
    ColumnEncoding,
    decode_dataframe,
    encode_dataframe,
    infer_categorical_columns,
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)
from katabatic.models.base_model import Model as BaseModel


class AIMModel(BaseModel):
    """
    Differentially private synthetic data generator using AIM.

    AIM adaptively selects and measures informative marginals under
    differential privacy, then estimates a Private-PGM graphical model
    from those noisy measurements.
    """

    ARTIFACT_STATE_FILES = ("aim_state.pkl",)

    def __init__(
        self,
        *,
        epsilon: float = 1.0,
        delta: float = 1e-9,
        degree: int = 2,
        max_bins: int = 10,
        max_cells: int = 10000,
        max_model_size: float = 80,
        max_iters: int = 1000,
        rounds: int | None = None,
        seed: int | None = 42,
        categorical_columns: list[str] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> None:
        super().__init__()

        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1.")

        if degree <= 0:
            raise ValueError("degree must be greater than zero.")

        if max_bins < 2:
            raise ValueError("max_bins must be at least 2.")

        if max_cells <= 0:
            raise ValueError("max_cells must be greater than zero.")

        if max_model_size <= 0:
            raise ValueError("max_model_size must be greater than zero.")

        if max_iters <= 0:
            raise ValueError("max_iters must be greater than zero.")

        if rounds is not None and rounds <= 0:
            raise ValueError("rounds must be positive when provided.")

        self.epsilon = epsilon
        self.delta = delta
        self.degree = degree
        self.max_bins = max_bins
        self.max_cells = max_cells
        self.max_model_size = max_model_size
        self.max_iters = max_iters
        self.rounds = rounds
        self.seed = seed

        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns

        self.model: Any | None = None
        self.column_names: list[str] | None = None
        self.label: str | None = None

        self._train_df: pd.DataFrame | None = None
        self._train_row_count: int | None = None
        self._encodings: dict[str, ColumnEncoding] = {}
        self._domain_sizes: dict[str, int] = {}
        self._resolved_categorical_columns: list[str] = []
        self._resolved_continuous_columns: list[str] = []

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return Python import names required by AIM."""

        return ["mbi"]

    @staticmethod
    def _validate_column_list(
        columns: list[str],
        df: pd.DataFrame,
        kind: str,
    ) -> None:
        """Validate user-provided column names."""

        unknown_columns = [column for column in columns if column not in df.columns]

        if unknown_columns:
            raise ValueError(
                f"{kind} columns were not found in training data: {unknown_columns}"
            )

    def _resolve_column_types(
        self,
        df: pd.DataFrame,
        *,
        categorical_cols: list[str] | None,
        continuous_cols: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        """
        Resolve categorical and continuous columns.

        Benchmark-provided train() arguments take precedence over
        constructor configuration. If neither is supplied, categorical
        columns are inferred from pandas dtypes and the remainder are
        treated as continuous.
        """

        categorical = (
            list(categorical_cols)
            if categorical_cols is not None
            else (
                list(self.categorical_columns)
                if self.categorical_columns is not None
                else None
            )
        )

        continuous = (
            list(continuous_cols)
            if continuous_cols is not None
            else (
                list(self.continuous_columns)
                if self.continuous_columns is not None
                else None
            )
        )

        if categorical is None and continuous is None:
            categorical = infer_categorical_columns(df)
            continuous = [column for column in df.columns if column not in categorical]

        elif categorical is None:
            assert continuous is not None

            self._validate_column_list(
                continuous,
                df,
                "Continuous",
            )

            categorical = [column for column in df.columns if column not in continuous]

        elif continuous is None:
            self._validate_column_list(
                categorical,
                df,
                "Categorical",
            )

            continuous = [column for column in df.columns if column not in categorical]

        self._validate_column_list(
            categorical,
            df,
            "Categorical",
        )

        self._validate_column_list(
            continuous,
            df,
            "Continuous",
        )

        overlap = set(categorical) & set(continuous)

        if overlap:
            raise ValueError(
                f"Columns cannot be both categorical and continuous: {sorted(overlap)}"
            )

        classified = set(categorical) | set(continuous)
        missing = set(df.columns) - classified

        if missing:
            raise ValueError(
                "Every training column must be classified as categorical "
                f"or continuous. Unspecified columns: {sorted(missing)}"
            )

        return categorical, continuous

    def _build_workload(
        self,
        domain,
    ) -> list[tuple[tuple[str, ...], float]]:
        """Build the marginal workload used by AIM."""

        columns = list(domain)

        if not columns:
            raise ValueError("Training data must contain at least one column.")

        degree = min(
            self.degree,
            len(columns),
        )

        cliques = list(
            itertools.combinations(
                columns,
                degree,
            )
        )

        cliques = [
            clique for clique in cliques if domain.size(clique) <= self.max_cells
        ]

        if not cliques:
            one_way = [
                (column,)
                for column in columns
                if domain.size((column,)) <= self.max_cells
            ]

            cliques = one_way

        if not cliques:
            raise ValueError(
                "No AIM workload marginals satisfy max_cells. "
                "Increase max_cells or reduce column cardinality."
            )

        return [(clique, 1.0) for clique in cliques]

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """Persist fitted AIM state for the Katabatic artifact pipeline."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save AIM state before training.")

        os.makedirs(artifact_state_dir, exist_ok=True)

        payload = {
            "config": {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "degree": self.degree,
                "max_bins": self.max_bins,
                "max_cells": self.max_cells,
                "max_model_size": self.max_model_size,
                "max_iters": self.max_iters,
                "rounds": self.rounds,
                "seed": self.seed,
                "categorical_columns": self.categorical_columns,
                "continuous_columns": self.continuous_columns,
            },
            "model": self.model,
            "column_names": self.column_names,
            "label": self.label,
            "encodings": self._encodings,
            "domain_sizes": self._domain_sizes,
            "resolved_categorical_columns": self._resolved_categorical_columns,
            "resolved_continuous_columns": self._resolved_continuous_columns,
            "train_row_count": self._train_row_count,
        }

        state_path = os.path.join(
            artifact_state_dir,
            self.ARTIFACT_STATE_FILES[0],
        )

        with open(state_path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load_from_ref(
        cls,
        store: ArtifactStore,
        ref: ModelRef,
    ) -> AIMModel:
        """Reload a fitted AIM model from a Katabatic artifact."""
        state_path = store.open_path(
            f"{ref.state_relpath}/{cls.ARTIFACT_STATE_FILES[0]}"
        )

        if not state_path.is_file():
            raise FileNotFoundError(
                f"No AIM state found at {state_path}. "
                "The model must be trained through the artifact pipeline."
            )

        with open(state_path, "rb") as fh:
            payload = pickle.load(fh)  # nosec B301: loading our own artifact

        instance = cls(**payload["config"])
        instance.model = payload["model"]
        instance.column_names = payload["column_names"]
        instance.label = payload["label"]
        instance._encodings = payload["encodings"]
        instance._domain_sizes = payload["domain_sizes"]
        instance._resolved_categorical_columns = payload["resolved_categorical_columns"]
        instance._resolved_continuous_columns = payload["resolved_continuous_columns"]
        instance._train_row_count = payload["train_row_count"]
        instance._train_df = None
        instance.is_fitted = True

        return instance

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> AIMModel:
        """
        Fit AIM using Katabatic split data and save synthetic output.

        Supports the benchmark interface arguments ``categorical_cols``
        and ``continuous_cols``.
        """

        try:
            from mbi import Dataset, Domain
        except ImportError as exc:
            raise ImportError(
                "Private-PGM is required by AIM. Install the AIM optional dependencies."
            ) from exc

        from katabatic.models.aim.mechanism import AIMMechanism

        df = load_training_data(data_dir)

        if df.empty:
            raise ValueError("Training data must not be empty.")

        if df.isnull().any().any():
            raise ValueError("AIM training data must not contain missing values.")

        self.column_names = df.columns.tolist()
        self.label = df.columns[-1]
        self._train_df = df.copy()
        self._train_row_count = len(df)

        categorical_cols = kwargs.get("categorical_cols")
        continuous_cols = kwargs.get("continuous_cols")

        categorical_columns, continuous_columns = self._resolve_column_types(
            df,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )

        self._resolved_categorical_columns = categorical_columns
        self._resolved_continuous_columns = continuous_columns

        (
            encoded_df,
            self._encodings,
            self._domain_sizes,
        ) = encode_dataframe(
            df,
            categorical_columns,
            continuous_columns,
            max_bins=self.max_bins,
        )

        domain = Domain(
            self.column_names,
            [self._domain_sizes[column] for column in self.column_names],
        )

        private_data = Dataset(
            encoded_df,
            domain,
        )

        workload = self._build_workload(domain)

        mechanism = AIMMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            seed=self.seed,
            rounds=self.rounds,
            max_model_size=self.max_model_size,
            max_iters=self.max_iters,
        )

        print(
            "[AIM] Initializing with "
            f"epsilon={self.epsilon}, "
            f"delta={self.delta}, "
            f"degree={self.degree}..."
        )

        self.model, synthetic_dataset = mechanism.run(
            private_data,
            workload,
            num_synth_rows=len(df),
        )

        self.is_fitted = True

        synthetic_encoded = synthetic_dataset.df

        if not isinstance(
            synthetic_encoded,
            pd.DataFrame,
        ):
            synthetic_encoded = pd.DataFrame(
                synthetic_encoded,
                columns=self.column_names,
            )

        synthetic_df = decode_dataframe(
            synthetic_encoded,
            self._encodings,
            self.column_names,
        ).reset_index(drop=True)

        synth_dir = resolve_synth_dir(
            synthetic_dir,
            data_dir,
            "aim",
        )

        x_path, y_path = save_synthetic_data(
            synthetic_df,
            self.label,
            synth_dir,
        )

        save_metadata(
            synth_dir=synth_dir,
            df=df,
            label=self.label,
            epsilon=self.epsilon,
            delta=self.delta,
            degree=self.degree,
            max_bins=self.max_bins,
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
            domain_sizes=self._domain_sizes,
            n_generated=len(synthetic_df),
        )

        artifact_state_dir = kwargs.get("artifact_state_dir")
        if artifact_state_dir:
            self._save_artifact_state(artifact_state_dir)

        print(f"[AIM] Synthetic data saved:\n  X -> {x_path}\n  y -> {y_path}")

        return self

    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> float:
        """Return a placeholder score for Katabatic pipeline compatibility."""

        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        return 0.0

    def sample(
        self,
        n: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate decoded synthetic records from the fitted AIM model."""

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call train() before sample().")

        if n is None:
            if self._train_row_count is None:
                raise RuntimeError("Training row count is unavailable.")

            n = self._train_row_count

        if n <= 0:
            raise ValueError("n must be greater than 0.")

        if self.column_names is None:
            raise RuntimeError("Column metadata is unavailable.")

        if not self._encodings:
            raise RuntimeError("Encoding metadata is unavailable.")

        synthetic_dataset = self.model.synthetic_data(rows=int(n))

        synthetic_encoded = synthetic_dataset.df

        if not isinstance(
            synthetic_encoded,
            pd.DataFrame,
        ):
            synthetic_encoded = pd.DataFrame(
                synthetic_encoded,
                columns=self.column_names,
            )

        return decode_dataframe(
            synthetic_encoded,
            self._encodings,
            self.column_names,
        ).reset_index(drop=True)
