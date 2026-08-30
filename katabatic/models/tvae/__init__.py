"""Module for TVAE model.

Wraps the official ctgan package's TVAE implementation
(https://github.com/sdv-dev/CTGAN), by Xu et al. (2019), applying the VAE
framework of Kingma & Welling (2013). See utils.py for adaptation notes.
"""
from .models import TVAEModel

__all__ = ["TVAEModel"]
