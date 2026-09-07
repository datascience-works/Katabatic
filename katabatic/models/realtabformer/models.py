from pathlib import Path
from typing import Any

import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from katabatic.models.realtabformer.utils import (
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)


class REaLTabFormerModel(BaseModel):
    """
    Experimental Katabatic wrapper for REaLTabFormer.

    This implementation supports regular, non-relational tabular
    datasets only.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 8,
        random_state: int = 1029,
        device: str = "cpu",
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 100,
        fit_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than 0."
            )

        if logging_steps <= 0:
            raise ValueError("logging_steps must be greater than 0.")

        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
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
        synthetic_dir: str | Path | None = None,
        *args,
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

        fit_kwargs = {"n_critic": 0, **self.fit_kwargs}

        self.model.fit(
          training_df,
          device=self.device,
          **fit_kwargs,
        )
        self.is_fitted = True

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
                "gradient_accumulation_steps": (
                    self.gradient_accumulation_steps
                ),
                "logging_steps": self.logging_steps,
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