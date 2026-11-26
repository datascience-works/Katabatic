"""Tests for the GReaT timing estimate and measurement."""

import pandas as pd
import pytest

from benchmarks import great_timing as gt


def _estimate(**overrides):
    values = {
        "seconds_per_epoch": 10.0,
        "timed_rows": 100,
        "full_rows": 1000,
        "rows_per_second": 10.0,
        "planned_epochs": 10,
        "load_seconds": 0.0,
        "sampling_passes": 1,
        "budget_seconds": 1000.0,
        "safety": 1.0,
    }
    values.update(overrides)
    return gt.estimate(**values)


def test_epoch_time_scales_with_rows():
    assert _estimate()["seconds_per_epoch_full"] == 100.0


def test_sampling_time_uses_rows_per_second_and_passes():
    result = _estimate(sampling_passes=3)

    assert result["seconds_per_sampling_pass"] == 100.0
    assert result["seconds_sampling_total"] == 300.0


def test_safe_epochs_fill_what_is_left_after_sampling():
    # budget 1000 - sampling 100 = 900 s, at 100 s per epoch
    assert _estimate()["safe_epochs"] == 9


def test_safety_factor_and_load_time_reduce_the_budget():
    result = _estimate(safety=0.5, load_seconds=100.0)

    # 1000 * 0.5 - 100 load - 100 sampling = 300 s
    assert result["safe_epochs"] == 3


def test_safe_epochs_never_negative():
    assert _estimate(budget_seconds=50.0)["safe_epochs"] == 0


def test_planned_run_is_checked_against_the_budget():
    assert _estimate(planned_epochs=9)["planned_fits_budget"] is True
    assert _estimate(planned_epochs=10)["planned_fits_budget"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"seconds_per_epoch": 0},
        {"rows_per_second": 0},
        {"timed_rows": 0},
        {"full_rows": 0},
    ],
)
def test_bad_numbers_are_rejected(bad):
    with pytest.raises(ValueError):
        _estimate(**bad)


class FakeModel:
    def __init__(self, clock):
        self.clock = clock
        self.sample_calls = []

    def fit(self, frame):
        self.clock.now += 20.0
        self.rows = len(frame)

    def sample(self, n, device="cpu", seed=None, k=100):
        self.sample_calls.append(n)
        self.clock.now += 4.0 if device != "cpu" or k != 8 else 40.0
        return pd.DataFrame({"a": range(n)})


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


def _split(tmp_path, rows=50):
    folder = tmp_path / "car"
    folder.mkdir()
    pd.DataFrame({"a": range(rows), "b": range(rows)}).to_csv(
        folder / "train_full.csv", index=False
    )


def test_measure_times_each_phase_separately(tmp_path):
    _split(tmp_path)
    clock = FakeClock()

    timings = gt.measure(
        "car",
        epochs=2,
        train_rows=10,
        sample_rows=8,
        device="mps",
        compare_cpu=True,
        split_root=str(tmp_path),
        model_factory=lambda _dir: FakeModel(clock),
        clock=clock,
    )

    assert timings["timed_rows"] == 10
    assert timings["full_rows"] == 50
    assert timings["seconds_fit"] == 20.0
    assert timings["seconds_per_epoch_timed"] == 10.0
    assert timings["seconds_sample"] == 4.0
    assert timings["rows_per_second"] == 2.0
    assert timings["seconds_sample_cpu_small_batch"] == 40.0


def test_measure_never_uses_more_rows_than_exist(tmp_path):
    _split(tmp_path, rows=5)
    clock = FakeClock()

    timings = gt.measure(
        "car",
        train_rows=500,
        split_root=str(tmp_path),
        model_factory=lambda _dir: FakeModel(clock),
        clock=clock,
    )

    assert timings["timed_rows"] == 5


def test_missing_split_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run any benchmark script"):
        gt.measure("car", split_root=str(tmp_path))


def test_script_epochs_reads_the_great_scripts():
    assert gt.script_epochs("adult") == 310
    assert gt.script_epochs("car") == 2


def test_a_warmup_sample_is_run_before_the_timed_one(tmp_path):
    _split(tmp_path)
    clock = FakeClock()
    model = FakeModel(clock)

    gt.measure(
        "car",
        sample_rows=30,
        split_root=str(tmp_path),
        model_factory=lambda _dir: model,
        clock=clock,
    )

    assert model.sample_calls == [gt.WARMUP_ROWS, 30]


def test_macos_settings_are_set_by_default():
    import os
    import sys

    if sys.platform == "darwin":
        assert os.environ.get("OMP_NUM_THREADS") is not None
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") is not None
