# katabatic/models/__init__.py
"""
Models package: exposes all implemented models (GANs, MedGAN, etc).
"""

from .base_model import Model
from .registry import ModelRegistry, get_model, list_models

__all__ = ['Model', 'ModelRegistry', 'get_model', 'list_models', 'TabDDPM']
