from __future__ import annotations

import pandas as pd
import pytest

from katabatic.models.mst import MSTModel


def test_mst_initial_state():
    model = MSTModel()

    assert model.is_fitted is False
    assert model.epsilon == 3.0
    assert model.delta is None


def test_mst_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon must be greater than 0"):
        MSTModel(epsilon=0)


def test_mst_invalid_delta():
    with pytest.raises(ValueError, match="delta must be between 0 and 1"):
        MSTModel(delta=1.0)


def test_mst_required_dependencies():
    assert MSTModel.get_required_dependencies() == [
        "snsynth",
        "mbi",
        "opendp",
    ]


def test_mst_sample_before_training():
    model = MSTModel()

    with pytest.raises(RuntimeError, match="Call train"):
        model.sample(10)


def test_mst_evaluate_before_training():
    model = MSTModel()

    with pytest.raises(RuntimeError, match="Call train"):
        model.evaluate()


def test_mst_infer_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "B", "C"],
            "active": [True, False, True],
        }
    )

    result = MSTModel._infer_categorical_columns(df)

    assert result == ["city", "active"]


def test_mst_resolve_default_delta():
    model = MSTModel()

    delta = model._resolve_delta(100)

    assert delta == pytest.approx(0.001)


def test_mst_resolve_explicit_delta():
    model = MSTModel(delta=0.0001)

    assert model._resolve_delta(100) == 0.0001
