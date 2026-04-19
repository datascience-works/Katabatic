from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    """Base class for all models in Katabatic."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def train(self, *args, categorical_cols: list = None, continuous_cols: list = None, **kwargs) -> 'Model':
        """
        Train the model on the given data.

        Parameters
        ----------
        categorical_cols : list[str], optional
            Column names to treat as categorical. When provided, the model
            uses these instead of its own internal type detection heuristic.
        continuous_cols : list[str], optional
            Column names to treat as continuous. When provided, the model
            uses these instead of its own internal type detection heuristic.
        """
        ...

    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate n_samples rows of synthetic data.

        CONTRACT — every implementation must honour all of the following:

        1. Return type: always a pd.DataFrame, never a numpy array or list.

        2. Columns: the returned DataFrame must contain exactly the same
           column names as the training data (same set, any order is fine).
           Do not add, remove, or rename columns.

        3. Data types: each column must have the same dtype as the
           corresponding training column, or a directly compatible one
           (e.g. int32 vs int64 is fine; float64 for a string categorical
           column is not — that means the model forgot to inverse-transform).

        4. Categorical values: string/categorical columns must contain values
           from the original training vocabulary. The model is responsible for
           inverse-transforming its internal integer/embedding codes back to
           the original strings before returning.

        5. No NaN columns: every column must have at least some valid values.

        Failing to honour this contract will cause the evaluation pipeline to
        raise a clear error via Model.validate_synthetic_output().
        """
        ...

    @staticmethod
    def validate_synthetic_output(
        synthetic_df: pd.DataFrame,
        real_df: pd.DataFrame,
        categorical_cols: list = None,
    ) -> None:
        """
        Validate that synthetic_df honours the sample() contract against real_df.

        Raises ValueError for hard violations that would corrupt evaluation.
        Prints warnings for soft violations that are worth knowing but won't
        break the pipeline.

        Parameters
        ----------
        synthetic_df : pd.DataFrame
            Output of model.sample().
        real_df : pd.DataFrame
            The training DataFrame the model was trained on.
        categorical_cols : list[str], optional
            Known categorical columns. Used to check type preservation.
            If omitted, inferred from real_df dtypes.
        """
        # --- Hard checks (raise) ---

        if not isinstance(synthetic_df, pd.DataFrame):
            raise ValueError(
                f"model.sample() must return a pd.DataFrame, "
                f"got {type(synthetic_df).__name__}."
            )

        real_cols  = set(real_df.columns)
        synth_cols = set(synthetic_df.columns)

        missing = real_cols - synth_cols
        if missing:
            raise ValueError(
                f"Synthetic data is missing columns that exist in the training data: {sorted(missing)}. "
                f"The model must return all original columns."
            )

        cat_cols = categorical_cols or [
            c for c in real_df.columns
            if not pd.api.types.is_numeric_dtype(real_df[c])
        ]

        for col in cat_cols:
            if col not in synthetic_df.columns:
                continue
            if pd.api.types.is_float_dtype(synthetic_df[col]):
                raise ValueError(
                    f"Column '{col}' is categorical in the training data but the model "
                    f"returned it as float. The model likely forgot to inverse-transform "
                    f"its internal encoding before returning from sample()."
                )

        # --- Soft checks (warn) ---

        extra = synth_cols - real_cols
        if extra:
            print(
                f"[WARNING] Synthetic data contains extra columns not in training data: {sorted(extra)}. "
                f"They will be ignored by the evaluation pipeline."
            )

        for col in cat_cols:
            if col not in synthetic_df.columns:
                continue
            real_vocab  = set(real_df[col].dropna().astype(str).unique())
            synth_vocab = set(synthetic_df[col].dropna().astype(str).unique())
            novel = synth_vocab - real_vocab
            if novel:
                print(
                    f"[WARNING] Column '{col}' contains values not seen in training data: "
                    f"{sorted(novel)[:5]}{'...' if len(novel) > 5 else ''}. "
                    f"This may indicate the model is not inverse-transforming correctly."
                )

        all_null_cols = [
            c for c in synthetic_df.columns if synthetic_df[c].isna().all()
        ]
        if all_null_cols:
            print(
                f"[WARNING] The following synthetic columns are entirely NaN: {all_null_cols}."
            )

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return a list of required dependencies for this model."""
        return []

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        required_deps = self.get_required_dependencies()
        missing_deps = []

        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for {self.__class__.__name__}: {missing_deps}. "
                f"Install with: pip install katabatic[{self.__class__.__name__.lower()}]"
            )

        return True
