"""Deprecated: use ``katabatic.utils.preprocess`` instead."""

import warnings

warnings.warn(
    "Import from katabatic.utils.preprocess instead of the repo-root utils module.",
    DeprecationWarning,
    stacklevel=2,
)

from katabatic.utils.preprocess import *  # noqa: F403
