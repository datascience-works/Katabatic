import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from katabatic.models.realtabformer.utils import (
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


class REaLTabFormerModel(BaseModel):
    """
    Katabatic wrapper for REaLTabFormer.

    This implementation supports regular, non-relational tabular
    datasets only.
    """

    ARTIFACT_STATE_FILES = ("realtabformer_state.json",)

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 8,
        random_state: int = 1029,
        device: str = "cpu",
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 100,
        n_critic: int = 5,
        fit_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be greater than 0.")

        if logging_steps <= 0:
            raise ValueError("logging_steps must be greater than 0.")

        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.n_critic = n_critic
        self.fit_kwargs = fit_kwargs or {}

        self.model = None
        self.target_column: str | None = None
        self.column_names: list[str] | None = None
        self.training_rows: int | None = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["realtabformer"]

    def train(
        self,
        data_dir: str | Path,
        *args,
        synthetic_dir: str | Path | None = None,
        artifact_state_dir: str | Path | None = None,
        **kwargs,
    ) -> "REaLTabFormerModel":
        """
        Train REaLTabFormer using Katabatic-formatted training data.

        Synthetic data with the same number of rows as the training data
        is generated after training and written to Katabatic's standard
        synthetic output files.
        """

        self.check_dependencies()

        from realtabformer import REaLTabFormer

        training_df = load_training_data(data_dir)

        if training_df.empty:
            raise ValueError("Training data must not be empty.")

        if training_df.columns.duplicated().any():
            raise ValueError("Training data contains duplicate column names.")

        if training_df.isnull().any().any():
            raise ValueError(
                "Training data contains missing values. "
                "REaLTabFormer training requires complete input data."
            )

        self.target_column = str(training_df.columns[-1])
        self.column_names = list(training_df.columns)
        self.training_rows = len(training_df)

        self.model = REaLTabFormer(
            model_type="tabular",
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=self.logging_steps,
            **kwargs,
        )
        self.model.training_args_kwargs.pop("overwrite_output_dir", None)

        # REaLTabFormer 0.2.4 defaults gen_kwargs to None, but its
        # sensitivity-training path expands it using **gen_kwargs.
        # Supplying an empty dict avoids a NoneType failure while
        # preserving the library's normal generation behaviour.
        fit_kwargs = dict(self.fit_kwargs)
        fit_kwargs.setdefault("gen_kwargs", {})
        fit_kwargs["n_critic"] = self.n_critic

        self.model.fit(
            training_df,
            device=self.device,
            **fit_kwargs,
        )

        self.is_fitted = True
        self._maybe_save_artifact_state(artifact_state_dir)

        generated_df = self.sample(self.training_rows)

        resolved_synth_dir = resolve_synth_dir(
            synthetic_dir,
            data_dir,
            "realtabformer",
        )

        save_synthetic_data(
            generated_df,
            resolved_synth_dir,
            self.target_column,
        )

        save_metadata(
            resolved_synth_dir,
            target_column=self.target_column,
            row_count=len(generated_df),
            model_parameters={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "device": self.device,
                "gradient_accumulation_steps": (self.gradient_accumulation_steps),
                "logging_steps": self.logging_steps,
                "n_critic": self.n_critic,
            },
        )

        return self

    def sample(
        self,
        n: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic rows from the trained REaLTabFormer model.
        """

        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "The REaLTabFormer model must be trained before sampling."
            )

        if n is None:
            if self.training_rows is None:
                raise RuntimeError("Training row count is unavailable.")

            n = self.training_rows

        if n <= 0:
            raise ValueError("n must be greater than 0.")

        # Katabatic's stability evaluator passes seed=<value> to sample().
        # REaLTabFormer does not accept seed as a generation argument, so
        # consume it here and seed the underlying random generators instead.
        seed = kwargs.pop("seed", None)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

            try:
                import torch

                torch.manual_seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            except ImportError:
                pass

        synthetic_df = self.model.sample(
            n_samples=n,
            device=self.device,
            *args,
            **kwargs,
        )

        if not isinstance(synthetic_df, pd.DataFrame):
            synthetic_df = pd.DataFrame(
                synthetic_df,
                columns=self.column_names,
            )

        return synthetic_df.reset_index(drop=True)

    def evaluate(self, *args, **kwargs) -> float:
        """
        Placeholder evaluation hook required by Katabatic's Model interface.

        Evaluation metrics are handled by Katabatic's benchmarking and
        evaluation pipeline rather than inside this model wrapper.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The REaLTabFormer model must be trained before evaluation."
            )

        return 0.0

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted REaLTabFormer state for the Katabatic artifact pipeline.
        """
        artifact_state_dir = Path(artifact_state_dir)
        artifact_state_dir.mkdir(parents=True, exist_ok=True)
        self.model.full_save_dir = str(self.model.full_save_dir)
        self.model.save(artifact_state_dir, allow_overwrite=True)

        state = {
            "experiment_id": self.model.experiment_id,
            "target_column": self.target_column,
            "column_names": self.column_names,
            "training_rows": self.training_rows,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "device": self.device,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "n_critic": self.n_critic,
            "fit_kwargs": self.fit_kwargs,
        }
        state_path = artifact_state_dir / self.ARTIFACT_STATE_FILES[0]
        state_path.write_text(json.dumps(state))

    @classmethod
    def load_from_ref(
        cls, store: "ArtifactStore", ref: "ModelRef"
    ) -> "REaLTabFormerModel":
        """Reload a fitted REaLTabFormer model from a Katabatic artifact reference."""
        from realtabformer import REaLTabFormer

        state_path = cls._require_state_file(store, ref)
        state = json.loads(state_path.read_text())

        instance = cls(
            epochs=state["epochs"],
            batch_size=state["batch_size"],
            random_state=state["random_state"],
            device=state["device"],
            gradient_accumulation_steps=state["gradient_accumulation_steps"],
            logging_steps=state["logging_steps"],
            n_critic=state["n_critic"],
            fit_kwargs=state["fit_kwargs"],
        )
        instance.target_column = state["target_column"]
        instance.column_names = state["column_names"]
        instance.training_rows = state["training_rows"]
        instance.model = REaLTabFormer.load_from_dir(
            state_path.parent / state["experiment_id"]
        )
        instance.is_fitted = True

        return instance
