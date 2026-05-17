"""
CTAB-GAN+ package initialisation for the Katabatic framework.

This file exposes the main CTABGANPlus model class so it can be imported
cleanly from the ctabgan_plus package during benchmarking and pipeline use.
"""

from katabatic.models.ctabgan_plus.models import CTABGANPlus

__all__ = ["CTABGANPlus"]