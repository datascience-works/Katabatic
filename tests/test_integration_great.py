"""Integration tests for GReaT (import/registry; full train is manual)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from katabatic.models.registry import ModelRegistry


@pytest.mark.integration
@pytest.mark.great
def test_great_registry_load():
    cls = ModelRegistry.load_model("great")
    assert cls.__name__ == "GReaT"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")
