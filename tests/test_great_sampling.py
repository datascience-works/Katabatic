"""Tests for benchmarks/great_sampling.py, using a fake GReaT model."""

import sys
from pathlib import Path

import pandas as pd
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import great_sampling  # noqa: E402
from great_sampling import SamplingBudgetExceeded, sample_in_chunks  # noqa: E402


class FakeModel:
    """Returns n rows per call and remembers the seeds it was given."""

    def __init__(self, short_after=None):
        self.calls = []
        self.short_after = short_after

    def sample(self, n, device, seed):
        self.calls.append((n, device, seed))
        if self.short_after is not None and len(self.calls) > self.short_after:
            return pd.DataFrame({"a": []})
        return pd.DataFrame({"a": range(n)})


class FakeClock:
    def __init__(self, step):
        self.now = -step
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def test_returns_exactly_the_requested_rows():
    model = FakeModel()

    result = sample_in_chunks(model, 4500, "cpu", seed=42, chunk_size=2000)

    assert len(result) == 4500
    assert [call[0] for call in model.calls] == [2000, 2000, 500]


def test_every_chunk_gets_its_own_seed_starting_at_the_given_one():
    model = FakeModel()

    sample_in_chunks(model, 5000, "cpu", seed=42, chunk_size=2000)

    assert [call[2] for call in model.calls] == [42, 43, 44]


def test_is_reproducible():
    first = sample_in_chunks(FakeModel(), 300, "cpu", seed=7, chunk_size=100)
    second = sample_in_chunks(FakeModel(), 300, "cpu", seed=7, chunk_size=100)

    pd.testing.assert_frame_equal(first, second)


def test_passes_the_device_through():
    model = FakeModel()

    sample_in_chunks(model, 10, "mps", seed=1)

    assert model.calls[0][1] == "mps"


def test_stops_when_the_time_limit_is_reached():
    model = FakeModel()

    with pytest.raises(SamplingBudgetExceeded, match="Known limitation"):
        sample_in_chunks(
            model,
            10000,
            "cpu",
            seed=42,
            chunk_size=1000,
            max_seconds=25,
            clock=FakeClock(step=10),
        )

    assert 1 < len(model.calls) < 10


def test_no_error_if_the_last_chunk_finishes_after_the_limit():
    result = sample_in_chunks(
        FakeModel(),
        1000,
        "cpu",
        seed=42,
        chunk_size=1000,
        max_seconds=5,
        clock=FakeClock(step=10),
    )

    assert len(result) == 1000


def test_no_limit_when_max_seconds_is_none():
    result = sample_in_chunks(
        FakeModel(),
        3000,
        "cpu",
        seed=1,
        chunk_size=1000,
        max_seconds=None,
        clock=FakeClock(step=1000),
    )

    assert len(result) == 3000


def test_empty_chunk_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="returned no rows after 1000 of 5000"):
        sample_in_chunks(FakeModel(short_after=1), 5000, "cpu", seed=1, chunk_size=1000)


def test_columns_are_applied():
    class Model:
        def sample(self, n, device, seed):
            return [[1, 2]] * n

    result = sample_in_chunks(Model(), 3, "cpu", seed=1, columns=["x", "y"])

    assert list(result.columns) == ["x", "y"]


def test_bad_arguments():
    with pytest.raises(ValueError):
        sample_in_chunks(FakeModel(), 0, "cpu", seed=1)
    with pytest.raises(ValueError):
        sample_in_chunks(FakeModel(), 10, "cpu", seed=1, chunk_size=0)


def test_defaults_are_one_hour_and_two_thousand_rows():
    assert great_sampling.DEFAULT_MAX_SECONDS == 3600
    assert great_sampling.DEFAULT_CHUNK_SIZE == 2000
