"""
Katabatic package initializer.
Synthetic tabular data generation, pipelines, and evaluation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("katabatic")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from . import models, pipeline, utils

__all__ = ["__version__", "models", "pipeline", "utils"]
