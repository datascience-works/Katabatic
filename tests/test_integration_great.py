"""Integration tests for GReaT (import/registry; full train is manual)."""

from __future__ import annotations

import pytest

from tests.conftest import require_backend

require_backend("torch", "save")
require_backend("transformers", "AutoModelForCausalLM")

from katabatic.models.registry import ModelRegistry  # noqa: E402


@pytest.mark.integration
@pytest.mark.great
def test_great_registry_load():
    cls = ModelRegistry.load_model("great")
    assert cls.__name__ == "GReaT"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")
