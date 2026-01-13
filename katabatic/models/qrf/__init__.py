"""
Quantile Regression Forest (QRF) model integration for Katabatic.

This module wraps the official `quantile-forest` library and provides
adapter and generator components for synthetic tabular data generation.
"""

from .models import QRFModel
from .adapter import QRFAdapter

__all__ = [
    "QRFModel",
    "QRFAdapter",
]
