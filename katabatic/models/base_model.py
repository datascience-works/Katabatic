from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np
import pandas as pd


class Model(ABC):
    """Base class for all models in Katabatic."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def train(self, *args, **kwargs) -> 'Model':
        """Train the model on the given data."""
        ...

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate the model performance."""
        ...

    @abstractmethod
    def sample(self, *args, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Generate synthetic samples."""
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
