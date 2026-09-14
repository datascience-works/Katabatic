# katabatic/models/__init__.py
"""
Models package: exposes the model base class and the registry.

Model classes are not imported here: each pulls in heavy optional dependencies
(TensorFlow, PyTorch), so they are loaded on demand via ``ModelRegistry``.
"""

from .base_model import Model
from .registry import ModelRegistry, get_model, list_models, list_supported_models

__all__ = [
    "Model",
    "ModelRegistry",
    "get_model",
    "list_models",
    "list_supported_models",
]
