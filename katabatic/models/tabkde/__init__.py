"""Module for TabKDE model.

Adapted from the official TabKDE repository (https://github.com/tabkde/tabkde-main),
authors of the TabKDE paper. Ported and modified to fit the Katabatic
model interface — see utils.py for adaptation notes.
"""
from .models import TabKDEModel

__all__ = ["TabKDEModel"]
