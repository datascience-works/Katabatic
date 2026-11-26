"""Unit tests for scripts/release_smoke_test.py (no network, no venv creation)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_smoke_test.py"
spec = importlib.util.spec_from_file_location("release_smoke_test", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_requirement_pins_version_and_extras():
    assert smoke.requirement("1.2.0rc1", []) == "katabatic==1.2.0rc1"
    assert (
        smoke.requirement("1.2.0", ["ctgan", "eval"]) == "katabatic[ctgan,eval]==1.2.0"
    )
    assert smoke.requirement(None, []) == "katabatic"


def test_testpypi_takes_package_from_testpypi_and_deps_from_pypi():
    args = smoke.install_args("testpypi", "1.2.0rc1", None, [])
    assert args[args.index("--index-url") + 1] == smoke.TESTPYPI_INDEX
    assert args[args.index("--extra-index-url") + 1] == smoke.PYPI_INDEX
    assert args[-1] == "katabatic==1.2.0rc1"


def test_pypi_uses_default_index():
    args = smoke.install_args("pypi", "1.2.0", None, [])
    assert "--index-url" not in args
    assert args[-1] == "katabatic==1.2.0"


def test_wheel_install_adds_extras_to_the_file_path(tmp_path):
    wheel = tmp_path / "katabatic-1.2.0-py3-none-any.whl"
    wheel.touch()
    assert smoke.install_args(None, None, str(wheel), []) == [str(wheel)]
    assert smoke.install_args(None, None, str(wheel), ["smote"]) == [f"{wheel}[smote]"]


def test_source_requires_version():
    with pytest.raises(SystemExit):
        smoke.parse_args(["--source", "testpypi"])


def test_source_and_wheel_are_mutually_exclusive(tmp_path):
    wheel = tmp_path / "k.whl"
    wheel.touch()
    with pytest.raises(SystemExit):
        smoke.parse_args(
            ["--source", "pypi", "--version", "1.2.0", "--wheel", str(wheel)]
        )


def test_missing_wheel_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        smoke.parse_args(["--wheel", str(tmp_path / "missing.whl")])


def test_extras_are_repeatable():
    args = smoke.parse_args(
        [
            "--source",
            "pypi",
            "--version",
            "1.2.0",
            "--extra",
            "ctgan",
            "--extra",
            "eval",
        ]
    )
    assert args.extra == ["ctgan", "eval"]


def test_summary_fails_when_any_check_fails(capsys):
    assert smoke.print_summary([{"name": "a", "ok": True, "detail": ""}])
    assert not smoke.print_summary(
        [
            {"name": "a", "ok": True, "detail": ""},
            {"name": "b", "ok": False, "detail": "boom"},
        ]
    )
    assert "FAIL" in capsys.readouterr().out
