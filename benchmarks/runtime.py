"""Runtime, memory and hardware reporting for benchmark runs.

``runner.py`` calls ``start_run`` at the start of ``preprocess_and_split`` and
``finish_run`` at the end of ``evaluate``, so every benchmark script prints the
same report and saves ``<model>_<dataset>_runtime.json`` next to its results,
without changing the script itself.

The JSON file records:

* the time spent in preprocessing, in the model (train and generate) and in
  evaluation, plus any extra phases a script times with ``phase(...)``;
* peak RAM, and GPU memory when a GPU is used;
* the machine, the Git commit and the package versions;
* a ``status``: ``completed``, ``failed`` (the script raised an error, which is
  stored) or ``incomplete`` (the script ended before evaluation finished).

A script can time its own steps, for example for GReaT::

    import runtime

    with runtime.phase("train"):
        model.train(...)
    with runtime.phase("generate"):
        synthetic_df = model.sample(n)
"""

import atexit
import json
import os
import platform
import subprocess  # nosec B404 - only used to read the Git commit
import sys
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from time import perf_counter

_TIMER = None
_HOOKS_INSTALLED = False
_PACKAGES = ("katabatic", "torch", "numpy", "pandas", "scikit-learn")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_hardware() -> dict:
    """Describe the machine. Never raises, even if torch or psutil are missing."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": None,
        "accelerator": "CPU only",
    }
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:  # noqa: BLE001 - reporting must never stop a run
        pass
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["accelerator"] = f"CUDA: {name} x{count} ({memory:.1f} GB)"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["accelerator"] = "Apple MPS"
    except Exception:  # noqa: BLE001
        pass
    return info


def package_versions() -> dict:
    """Installed versions of the packages that decide the results."""
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            versions[name] = None
    return versions


def git_state() -> dict:
    """Short Git commit and whether the working tree has uncommitted changes."""
    state = {"commit": None, "dirty": None}
    try:
        base = ["git", "-C", _REPO_ROOT]
        commit = subprocess.run(  # nosec B603 B607
            base + ["rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            state["commit"] = commit.stdout.strip()
            changes = subprocess.run(  # nosec B603 B607
                base + ["status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            state["dirty"] = bool(changes.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return state


def peak_memory() -> dict:
    """Peak RAM of this process and GPU memory, in GB. Missing values are None."""
    peak = {"peak_ram_gb": None, "gpu_memory_gb": None}
    try:
        import resource

        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes
        scale = 1 if sys.platform == "darwin" else 1024
        peak["peak_ram_gb"] = round(used * scale / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            peak["gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            peak["gpu_memory_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return peak


class RunTimer:
    """Times a benchmark run and records how it ended."""

    def __init__(self, model: str, dataset: str, clock=perf_counter):
        self.model = model
        self.dataset = dataset
        self.output_dir = None
        self.status = "incomplete"
        self.error = None
        self._clock = clock
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.ended_at = None
        self._t0 = clock()
        self._total = None
        self._phases = {}
        self._open = {}

    def begin(self, phase: str) -> None:
        self._open[phase] = self._clock()

    def end(self, phase: str) -> None:
        started = self._open.pop(phase, None)
        if started is not None:
            self._phases[phase] = self._phases.get(phase, 0.0) + self._clock() - started

    def finish(self, status: str | None = None) -> None:
        self._total = self._clock() - self._t0
        self.ended_at = datetime.now().isoformat(timespec="seconds")
        if status:
            self.status = status

    @property
    def finished(self) -> bool:
        return self._total is not None

    def summary(self) -> dict:
        total = self._total if self._total is not None else self._clock() - self._t0
        preprocess = self._phases.get("preprocess", 0.0)
        evaluate = self._phases.get("evaluate", 0.0)
        extra = {
            name: round(seconds, 2)
            for name, seconds in self._phases.items()
            if name not in ("preprocess", "evaluate")
        }
        data = {
            "model": self.model,
            "dataset": self.dataset,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_seconds": round(total, 2),
            "preprocess_seconds": round(preprocess, 2),
            "model_seconds": round(max(total - preprocess - evaluate, 0.0), 2),
            "evaluate_seconds": round(evaluate, 2),
            "phases": extra,
        }
        data.update(peak_memory())
        data["git"] = git_state()
        data["packages"] = package_versions()
        data["hardware"] = detect_hardware()
        return data

    def print_report(self) -> dict:
        data = self.summary()
        hw = data["hardware"]
        line = "=" * 70
        print(f"\n{line}\nEvaluation Runtime Report\n{line}")
        print(f"Start time : {data['started_at']}")
        print(f"End time   : {data['ended_at']}")
        print(f"{data['model']} on {data['dataset']}   status: {data['status']}")
        print(f"  Total                 : {data['total_seconds']} s")
        print(f"  Preprocess and split  : {data['preprocess_seconds']} s")
        print(f"  Train and generate    : {data['model_seconds']} s")
        for name, seconds in data["phases"].items():
            print(f"    {name:<20}: {seconds} s")
        print(f"  Evaluate              : {data['evaluate_seconds']} s")
        gpu = data["gpu_memory_gb"]
        print(f"  Peak RAM              : {data['peak_ram_gb']} GB")
        print(f"  GPU memory            : {'n/a' if gpu is None else f'{gpu} GB'}")
        print(f"{line}\nComputation Hardware Summary\n{line}")
        print(f"  System      : {hw['system']} {hw['release']} ({hw['machine']})")
        print(f"  Processor   : {hw['processor']} ({hw['cpu_count']} cores)")
        print(f"  Accelerator : {hw['accelerator']}")
        print(f"  Total RAM   : {hw['ram_total_gb']} GB")
        print(f"  Python      : {hw['python']}")
        print(
            f"  Git commit  : {data['git']['commit']} (uncommitted changes: {data['git']['dirty']})"
        )
        print(line)
        return data

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.model}_{self.dataset}_runtime.json")
        with open(path, "w") as handle:
            json.dump(self.summary(), handle, indent=2)
        return path


def _write_unfinished() -> None:
    """Save a record for a run that did not reach the end of evaluate."""
    timer = _TIMER
    if timer is None or timer.finished or not timer.output_dir:
        return
    try:
        timer.finish("failed" if timer.error else "incomplete")
        timer.save(timer.output_dir)
    except Exception:  # noqa: BLE001 - never raise while the interpreter exits
        pass


def _install_hooks() -> None:
    """Remember an uncaught error and save an unfinished run when Python exits."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if _TIMER is not None and not _TIMER.finished:
            _TIMER.error = f"{exc_type.__name__}: {exc}"
        previous(exc_type, exc, tb)

    sys.excepthook = hook
    atexit.register(_write_unfinished)


def start_run(model: str, dataset: str) -> RunTimer:
    """Start the timer for this run and make it the current one."""
    global _TIMER
    _TIMER = RunTimer(model, dataset)
    _install_hooks()
    return _TIMER


def current() -> RunTimer | None:
    return _TIMER


@contextmanager
def phase(name: str):
    """Time a block of a script: ``with runtime.phase("train"): ...``."""
    timer = _TIMER
    if timer is None:
        yield
        return
    timer.begin(name)
    try:
        yield
    finally:
        timer.end(name)


def finish_run(output_dir: str) -> dict | None:
    """Stop the timer, print the report and save the JSON file."""
    if _TIMER is None:
        return None
    _TIMER.finish("completed")
    data = _TIMER.print_report()
    try:
        path = _TIMER.save(output_dir)
        print(f"Runtime saved to: {path}")
    except OSError as exc:
        print(f"WARNING: could not save the runtime file: {exc}")
    return data
