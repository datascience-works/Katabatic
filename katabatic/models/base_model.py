from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


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

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float | dict[str, float]:
        """Evaluate the fitted model and return a score, or a dict of named scores."""
        ...

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
