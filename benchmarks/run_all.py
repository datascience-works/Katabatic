"""Run the benchmark scripts for every model and dataset, one after another.

Examples (from the repository root)::

    python benchmarks/run_all.py --dry-run
    python benchmarks/run_all.py --models histogram,kde --datasets car
    python benchmarks/run_all.py --datasets car,nursery
    python benchmarks/run_all.py --retry-failed
    python benchmarks/run_all.py --report

How it works
------------
* Every ``benchmarks/examples/<model>/run_<model>_<dataset>.py`` is one job.
* A job that already has a valid evaluation report is skipped, so the run can
  be stopped and started again at any time. ``--force`` runs it anyway.
* Each job runs in its own process, so one crash never stops the others.
* A job that is still running after ``--timeout`` seconds (default 2 hours,
  the stream's 2-hour rule) is stopped and recorded as ``timeout``. This is a
  known limitation of the model, not a failure.
* Output of every job goes to ``benchmarks/results/run_all/logs/``.
* The outcome of every job is written to
  ``benchmarks/results/run_all/run_all_status.json`` after each job, and a
  Markdown summary is written to ``run_all_summary.md`` next to it.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.coverage import (  # noqa: E402
    DATASETS,
    SUCCESS,
    get_cell_status,
    get_models,
    read_json,
)

EXAMPLES_DIR = BENCHMARKS_DIR / "examples"
RESULTS_DIR = BENCHMARKS_DIR / "results"

DEFAULT_TIMEOUT = 2 * 60 * 60  # the 2-hour rule

# Outcome of one job
DONE = "completed"  # the script ran and finished
SKIPPED = "skipped"  # a valid report already exists
FAILED = "failed"
TIMEOUT = "timeout"  # stopped after --timeout seconds (known limitation)
PLANNED = "planned"  # dry run only

# Smallest datasets first, so quick results come before the slow ones
RUN_ORDER = ["car", "nursery", "magic", "adult", "shuttle"]


def split_names(text: str | None) -> list[str] | None:
    """Turn ``"a, b"`` into ``["a", "b"]``; ``None`` or empty means no filter."""
    if not text:
        return None
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    return names or None


def script_path(model: str, dataset: str, examples_dir: Path = EXAMPLES_DIR) -> Path:
    return examples_dir / model / f"run_{model}_{dataset}.py"


def build_jobs(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    exclude: list[str] | None = None,
    examples_dir: Path = EXAMPLES_DIR,
) -> list[tuple[str, str]]:
    """Return the (model, dataset) pairs to run, datasets smallest first.

    Raises ValueError for a model or dataset name that does not exist, so a
    typo never turns into an empty run.
    """
    available = get_models(examples_dir)
    for name in (models or []) + (exclude or []):
        if name not in available:
            raise ValueError(f"unknown model '{name}'. Models: {', '.join(available)}")
    for name in datasets or []:
        if name not in DATASETS:
            raise ValueError(
                f"unknown dataset '{name}'. Datasets: {', '.join(DATASETS)}"
            )

    wanted_models = [m for m in available if (not models or m in models)]
    wanted_models = [m for m in wanted_models if m not in (exclude or [])]
    wanted_datasets = [d for d in RUN_ORDER if (not datasets or d in datasets)]

    return [
        (model, dataset)
        for dataset in wanted_datasets
        for model in wanted_models
        if script_path(model, dataset, examples_dir).exists()
    ]


def result_dir(model: str, dataset: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / dataset / model


def previous_outcome(
    model: str,
    dataset: str,
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Coverage status of an earlier run: Success, Failed or Untested."""
    return get_cell_status(model, dataset, examples_dir, results_dir)


def should_run(
    model: str,
    dataset: str,
    force: bool = False,
    retry_failed: bool = False,
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> tuple[bool, str]:
    """Decide whether a job runs. Returns (run, reason)."""
    if force:
        return True, "forced"
    status = previous_outcome(model, dataset, examples_dir, results_dir)
    if status == SUCCESS:
        return False, "already has a valid report"
    if status == "Failed" and not retry_failed:
        return False, "failed before (use --retry-failed to run it again)"
    return True, "no valid report yet" if status != "Failed" else "retrying a failure"


def tail(path: Path, lines: int = 12) -> str:
    """Last lines of a log file, for the failure summary."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def run_job(
    model: str,
    dataset: str,
    timeout: float,
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
    logs_dir: Path | None = None,
    python: str = sys.executable,
) -> dict:
    """Run one benchmark script in its own process and describe what happened."""
    logs_dir = logs_dir or results_dir / "run_all" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{model}_{dataset}.log"
    script = script_path(model, dataset, examples_dir)
    runtime_path = result_dir(model, dataset, results_dir) / (
        f"{model}_{dataset}_runtime.json"
    )

    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    status = DONE
    error = None
    return_code = None

    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                [python, str(script)],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            status = TIMEOUT
            error = f"stopped after {timeout:.0f} s (known limitation, not a failure)"

    seconds = round(time.time() - started, 1)

    # The runtime file written by benchmarks/runtime.py tells more than the
    # exit code, but only if this run wrote it.
    runtime = None
    if runtime_path.exists() and runtime_path.stat().st_mtime >= started:
        runtime = read_json(runtime_path)

    if status != TIMEOUT:
        if return_code != 0:
            status = FAILED
            error = (
                (runtime or {}).get("error")
                or tail(log_path)
                or (f"exit code {return_code}")
            )
        elif runtime and runtime.get("status") in ("failed", "incomplete"):
            status = FAILED
            error = runtime.get("error") or f"runtime status {runtime['status']}"

    return {
        "model": model,
        "dataset": dataset,
        "status": status,
        "error": error,
        "return_code": return_code,
        "started_at": started_at,
        "seconds": seconds,
        "log": str(log_path),
        "peak_ram_gb": (runtime or {}).get("peak_ram_gb"),
    }


def write_status(records: list[dict], out_dir: Path) -> None:
    """Write the JSON ledger and the Markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_all_status.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    (out_dir / "run_all_summary.md").write_text(
        render_summary(records), encoding="utf-8"
    )


def render_summary(records: list[dict]) -> str:
    """Markdown summary: counts, then failures and timeouts with the reason."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    total_seconds = sum(record.get("seconds") or 0 for record in records)

    lines = ["# Benchmark run summary", ""]
    lines.append(f"{len(records)} jobs, {total_seconds / 60:.1f} minutes of run time.")
    lines.append("")
    lines.append("| Status | Jobs |")
    lines.append("|---|---|")
    for status in (DONE, SKIPPED, FAILED, TIMEOUT, PLANNED):
        if status in counts:
            lines.append(f"| {status} | {counts[status]} |")

    problems = [r for r in records if r["status"] in (FAILED, TIMEOUT)]
    if problems:
        lines += ["", "## Failed and timed out", ""]
        lines.append("| Model | Dataset | Status | Minutes | Reason |")
        lines.append("|---|---|---|---|---|")
        for record in problems:
            reason = (record.get("error") or "").strip().splitlines()
            reason = reason[-1] if reason else ""
            reason = reason.replace("|", "/")[:120]
            minutes = (record.get("seconds") or 0) / 60
            lines.append(
                f"| {record['model']} | {record['dataset']} | {record['status']} "
                f"| {minutes:.1f} | {reason} |"
            )
        lines += ["", "Full output of each job is in the log file named in the JSON."]
    return "\n".join(lines) + "\n"


def merge_records(old: list[dict], new: list[dict]) -> list[dict]:
    """Keep earlier outcomes; a newer outcome for the same job replaces it."""
    merged = {(r["model"], r["dataset"]): r for r in old}
    for record in new:
        if (
            record["status"] in (SKIPPED, PLANNED)
            and (
                record["model"],
                record["dataset"],
            )
            in merged
        ):
            continue
        merged[(record["model"], record["dataset"])] = record
    return [merged[key] for key in sorted(merged)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark scripts for many models and datasets."
    )
    parser.add_argument("--models", help="comma-separated models (default: all)")
    parser.add_argument("--datasets", help="comma-separated datasets (default: all)")
    parser.add_argument("--exclude", help="comma-separated models to leave out")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would run and what would be skipped, run nothing",
    )
    parser.add_argument(
        "--force", action="store_true", help="run jobs even if they have a report"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="also run jobs whose last run failed",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds before a job is stopped (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the summary of the last run and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = RESULTS_DIR / "run_all"
    status_path = out_dir / "run_all_status.json"

    if args.report:
        records = []
        if status_path.exists():
            with status_path.open(encoding="utf-8") as handle:
                records = json.load(handle)
        print(render_summary(records) if records else "No run recorded yet.")
        return 0

    try:
        jobs = build_jobs(
            split_names(args.models),
            split_names(args.datasets),
            split_names(args.exclude),
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    plan = []
    for model, dataset in jobs:
        run, reason = should_run(model, dataset, args.force, args.retry_failed)
        plan.append((model, dataset, run, reason))

    to_run = [item for item in plan if item[2]]
    print(
        f"{len(jobs)} jobs selected: {len(to_run)} to run, "
        f"{len(jobs) - len(to_run)} skipped."
    )

    if args.dry_run:
        for model, dataset, run, reason in plan:
            mark = "RUN " if run else "skip"
            print(f"  {mark} {model:<18} {dataset:<8} {reason}")
        return 0

    old = []
    if status_path.exists():
        with status_path.open(encoding="utf-8") as handle:
            old = json.load(handle)

    records: list[dict] = []
    for index, (model, dataset, run, reason) in enumerate(plan, start=1):
        if not run:
            records.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "status": SKIPPED,
                    "error": None,
                    "seconds": 0,
                    "reason": reason,
                }
            )
            continue
        print(f"[{index}/{len(plan)}] {model} on {dataset} ...", flush=True)
        record = run_job(model, dataset, args.timeout)
        records.append(record)
        print(f"    {record['status']} in {record['seconds']} s", flush=True)
        write_status(merge_records(old, records), out_dir)

    write_status(merge_records(old, records), out_dir)
    print()
    print(render_summary(records))
    failed = [r for r in records if r["status"] == FAILED]
    return 1 if failed else 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main())
