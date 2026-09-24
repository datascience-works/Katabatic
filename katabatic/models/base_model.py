from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef

EVALUATION_DIMENSIONS = (
    "fidelity",
    "utility",
    "diversity",
    "privacy",
    "consistency",
    "stability",
)


class EvaluationPipeline(Protocol):
    """
    ``Model.evaluate(pipeline=...)`` accepts anything with this ``run()``.

    ``SyntheticEvaluationPipeline`` is the default. A custom pipeline receives
    every argument by keyword; accept ``**kwargs`` to ignore the ones it doesn't use.
    """

    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str | None = None,
        test_data: pd.DataFrame | None = None,
        constraints: dict | None = None,
        model: Model | None = None,
        output_dir: str | None = None,
        report_prefix: str = "",
    ) -> Any: ...


class Model(ABC):
    """Base class for all models in Katabatic."""

    ARTIFACT_STATE_FILES: tuple[str, ...] = ()

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> Model:
        """
        Train the model on the given data.

        Parameters
        ----------
        data_dir : str
            Directory containing the model's expected training inputs.
        synthetic_dir : str, optional
            Where to write the generated ``x_synth.csv`` / ``y_synth.csv``.
            Defaults to ``synthetic/<dataset_name>/<model_slug>/`` when omitted.
        artifact_state_dir : str, optional
            When provided, fitted state is persisted here via
            ``_save_artifact_state`` for later reload through ``load_from_ref``.

        Returns
        -------
        Model
            ``self``, fitted, to allow chaining (e.g. ``Model().train(...)``).
        """
        ...

    def evaluate(
        self,
        real_data: pd.DataFrame,
        *,
        pipeline: EvaluationPipeline | None = None,
        target_col: str | None = None,
        test_data: pd.DataFrame | None = None,
        synthetic_data: pd.DataFrame | None = None,
        dimensions: list[str] | tuple[str, ...] | None = None,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
        constraints: dict | None = None,
        output_dir: str | None = None,
        report_prefix: str = "",
        **pipeline_kwargs,
    ) -> Any:
        """
        Score the fitted model's synthetic data.

        By default this runs
        :class:`katabatic.pipeline.evaluation_pipeline.SyntheticEvaluationPipeline`
        on all six dimensions (fidelity, utility via TSTR, diversity, privacy,
        consistency, stability), the same scoring the benchmark scripts use.
        Pass ``pipeline=`` to score with your own evaluation instead.

        Parameters
        ----------
        real_data : pd.DataFrame
            Real training data, target column included.
        pipeline : EvaluationPipeline, optional
            Any object with a ``run()`` method matching :class:`EvaluationPipeline`.
            Configure it before passing it in; ``dimensions``, ``categorical_cols``,
            ``continuous_cols`` and ``**pipeline_kwargs`` only apply to the default.
        target_col : str, optional
            Target column; defaults to the last column of ``real_data``.
        test_data : pd.DataFrame, optional
            Held-out real data. When given, utility's TSTR and TRTR test on it.
        synthetic_data : pd.DataFrame, optional
            Data to score. Defaults to ``self.sample(len(real_data))``.
        dimensions : list[str], optional
            Default pipeline only: subset of dimensions to run (default: all six).
            ``stability`` re-samples the model several times with different seeds.
        categorical_cols, continuous_cols : list[str], optional
            Default pipeline only: column type hints; auto-detected when omitted.
        constraints : dict, optional
            Per-column ``(min, max)`` bounds for the consistency dimension.
        output_dir : str, optional
            When given, the report is saved here.
        **pipeline_kwargs
            Default pipeline only: passed to ``SyntheticEvaluationPipeline``
            (e.g. ``weights``, ``n_folds``).

        Returns
        -------
        Whatever the pipeline's ``run()`` returns. For the default pipeline that
        is an ``EvaluationReport`` with ``composite_score``, ``dimension_scores``
        (one score per dimension) and ``dimension_results`` (full detail).
        """
        from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline

        if pipeline is not None:
            default_only = {
                "dimensions": dimensions,
                "categorical_cols": categorical_cols,
                "continuous_cols": continuous_cols,
                **pipeline_kwargs,
            }
            ignored = [k for k, v in default_only.items() if v is not None]
            if ignored:
                raise TypeError(
                    f"{ignored} configure the default pipeline and would be ignored "
                    "when pipeline= is given; configure your pipeline instead."
                )
            if not callable(getattr(pipeline, "run", None)):
                raise TypeError("pipeline must have a run() method.")

        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        if synthetic_data is None:
            synthetic_data = self.sample(len(real_data))
            if isinstance(synthetic_data, np.ndarray):
                synthetic_data = pd.DataFrame(synthetic_data, columns=real_data.columns)

        missing = [c for c in real_data.columns if c not in synthetic_data.columns]
        if missing:
            raise ValueError(f"Synthetic data is missing real columns: {missing}")
        synthetic_data = synthetic_data[list(real_data.columns)]

        if pipeline is None:
            pipeline = SyntheticEvaluationPipeline(
                dimensions=list(dimensions or EVALUATION_DIMENSIONS),
                categorical_cols=categorical_cols,
                continuous_cols=continuous_cols,
                **pipeline_kwargs,
            )
        return pipeline.run(
            real_data=real_data,
            synthetic_data=synthetic_data,
            target_col=target_col,
            test_data=test_data,
            constraints=constraints,
            model=self,
            output_dir=output_dir,
            report_prefix=report_prefix,
        )

    @abstractmethod
    def sample(
        self, n_samples: int | None = None, *args, **kwargs
    ) -> np.ndarray | pd.DataFrame:
        """
        Generate synthetic samples.

        Parameters
        ----------
        n_samples : int, optional
            Number of rows to generate. When omitted, models default to the
            number of rows they were trained on.
        """
        ...

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

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> Model:
        """
        Reload a fitted model from a versioned on-disk artifact (see ``ModelRef`` paths under the store).
        Subclasses may implement; default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{cls.__name__}.load_from_ref is not implemented for this model."
        )

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted state under ``artifact_state_dir`` to be rebuilt by
        load_from_ref().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._save_artifact_state is not implemented for this model."
        )

    def _maybe_save_artifact_state(self, artifact_state_dir: str | None) -> None:
        """
        Call from train() once fitting is complete. No operation when the caller doesn't request artifact persistence.
        """
        if artifact_state_dir:
            self._save_artifact_state(artifact_state_dir)

    @classmethod
    def _require_state_file(
        cls, store: ArtifactStore, ref: ModelRef, filename: str | None = None
    ) -> Path:
        """
        Resolve a single-file artifact state path under ref.state_relpath.
        """
        name = filename or cls.ARTIFACT_STATE_FILES[0]
        state_path = store.open_path(f"{ref.state_relpath}/{name}")
        if not state_path.is_file():
            raise FileNotFoundError(
                f"No {cls.__name__} artifact state at {state_path}. The model must "
                f"be trained through a pipeline that passes artifact_state_dir."
            )
        return state_path

    @classmethod
    def _require_state_dir(cls, store: ArtifactStore, ref: ModelRef) -> Path:
        """
        Resolve a multi-file artifact state directory under ref.state_relpath.
        """
        state_dir = store.open_path(ref.state_relpath)
        if not state_dir.is_dir():
            raise FileNotFoundError(
                f"{cls.__name__} artifact state directory not found: {ref.state_relpath}"
            )
        for name in cls.ARTIFACT_STATE_FILES:
            if not (state_dir / name).is_file():
                raise FileNotFoundError(
                    f"Missing {cls.__name__} artifact state file: {name}"
                )
        return state_dir
