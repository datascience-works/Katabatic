"""Integration tests for GReaT (import/registry; full train is manual)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from katabatic.models.registry import ModelRegistry


@pytest.mark.integration
@pytest.mark.great
def test_great_registry_load():
    assert ModelRegistry.is_supported("great")
    cls = ModelRegistry.load_model("great")
    assert cls.__name__ == "GReaT"
    instance = cls()
    assert hasattr(instance, "train")
    assert hasattr(instance, "sample")


@pytest.mark.integration
@pytest.mark.great
def test_supported_models_list():
    supported = ModelRegistry.get_supported_models()
    assert "ganblr" in supported
    assert "great" in supported
    assert "tabsyn" not in supported
