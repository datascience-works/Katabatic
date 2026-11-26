"""Time GReaT's training, sampling and evaluation passes separately.

GReaT can run for hours on the larger datasets. This tool measures a short
training run and a short sampling run on one dataset, then scales them up to
the full dataset, so the stream can write down a setting that fits the
2-hour rule (benchmarks/protocol.md) before starting a long run.

Example (after any benchmark has created benchmarks/splits/car/)::

    python benchmarks/great_timing.py --dataset car --epochs 1 --train-rows 500

The model and device are only used by ``measure``; ``estimate`` is plain
arithmetic and is tested without a language model.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from time import perf_counter

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same macOS settings as benchmarks/run_all.py. Without them, torch models can
# crash with a segmentation fault when two OpenMP runtimes are loaded.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

WARMUP_ROWS = 8

TWO_HOURS = 7200.0
SAFETY = 0.8  # use at most 80% of the budget, the rest is evaluation and noise
# One sampling pass for the saved data plus the stability evaluation's runs.
DEFAULT_SAMPLING_PASSES = 1 + 5
# Batch sizes used by benchmarks/examples/great/run_great_<dataset>.py
SCRIPT_BATCH_SIZE = {"adult": 128}
DEFAULT_BATCH_SIZE = 2


def estimate(
    seconds_per_epoch: float,
    timed_rows: int,
    full_rows: int,
    rows_per_second: float,
    planned_epochs: int,
    load_seconds: float = 0.0,
    sampling_passes: int = DEFAULT_SAMPLING_PASSES,
    budget_seconds: float = TWO_HOURS,
    safety: float = SAFETY,
) -> dict:
    """Scale short timings up to a full run and find the epochs that fit."""
    for name, value in (
        ("seconds_per_epoch", seconds_per_epoch),
        ("rows_per_second", rows_per_second),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be greater than 0")
    if timed_rows <= 0 or full_rows <= 0:
        raise ValueError("row counts must be greater than 0")

    epoch_seconds = seconds_per_epoch * full_rows / timed_rows
    pass_seconds = full_rows / rows_per_second
    sampling_seconds = pass_seconds * sampling_passes
    usable = budget_seconds * safety - load_seconds - sampling_seconds
    safe_epochs = max(0, math.floor(usable / epoch_seconds))
    planned_total = load_seconds + planned_epochs * epoch_seconds + sampling_seconds

    return {
        "seconds_per_epoch_full": round(epoch_seconds, 1),
        "seconds_per_sampling_pass": round(pass_seconds, 1),
        "sampling_passes": sampling_passes,
        "seconds_sampling_total": round(sampling_seconds, 1),
        "seconds_load": round(load_seconds, 1),
        "planned_epochs": planned_epochs,
        "seconds_planned_total": round(planned_total, 1),
        "planned_fits_budget": planned_total <= budget_seconds * safety,
        "safe_epochs": safe_epochs,
        "budget_seconds": budget_seconds,
        "safety": safety,
    }


def measure(
    dataset: str,
    epochs: int = 1,
    train_rows: int = 500,
    sample_rows: int = 100,
    device: str = "cpu",
    batch_size: int | None = None,
    llm: str = "gpt2",
    seed: int = 42,
    compare_cpu: bool = False,
    split_root: str | None = None,
    model_factory=None,
    clock=perf_counter,
) -> dict:
    """Run a short fit and a short sample and return the raw timings."""
    split_root = split_root or os.path.join(REPO_ROOT, "benchmarks", "splits")
    path = os.path.join(split_root, dataset, "train_full.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run any benchmark script for '{dataset}' once "
            "so the split exists."
        )
    full = pd.read_csv(path)
    timed = full.sample(n=min(train_rows, len(full)), random_state=seed)
    batch_size = batch_size or SCRIPT_BATCH_SIZE.get(dataset, DEFAULT_BATCH_SIZE)

    if model_factory is None:
        sys.path.insert(0, REPO_ROOT)
        from katabatic.models.great.models import GReaT

        def model_factory(experiment_dir):
            return GReaT(
                llm=llm,
                experiment_dir=experiment_dir,
                epochs=epochs,
                batch_size=batch_size,
            )

    with tempfile.TemporaryDirectory() as experiment_dir:
        start = clock()
        model = model_factory(experiment_dir)
        load_seconds = clock() - start

        start = clock()
        model.fit(timed)
        fit_seconds = clock() - start

        # The first call on a device compiles kernels and is much slower than
        # the calls after it, so it is run once without timing.
        model.sample(WARMUP_ROWS, device=device, seed=seed)

        start = clock()
        sampled = model.sample(sample_rows, device=device, seed=seed)
        sample_seconds = clock() - start

        cpu_seconds = None
        if compare_cpu:
            start = clock()
            model.sample(sample_rows, device="cpu", k=8, seed=seed)
            cpu_seconds = clock() - start

    return {
        "dataset": dataset,
        "device": device,
        "llm": llm,
        "batch_size": batch_size,
        "epochs_timed": epochs,
        "timed_rows": len(timed),
        "full_rows": len(full),
        "seconds_load": load_seconds,
        "seconds_fit": fit_seconds,
        "seconds_per_epoch_timed": fit_seconds / epochs,
        "sample_rows_asked": sample_rows,
        "sample_rows_returned": len(sampled),
        "seconds_sample": sample_seconds,
        "rows_per_second": len(sampled) / sample_seconds if sample_seconds else 0.0,
        "seconds_sample_cpu_small_batch": cpu_seconds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=1, help="epochs to time")
    parser.add_argument("--train-rows", type=int, default=500)
    parser.add_argument("--sample-rows", type=int, default=300)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--llm", default="gpt2")
    parser.add_argument(
        "--planned-epochs",
        type=int,
        help="epochs the script uses (default: read from the script)",
    )
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--budget", type=float, default=TWO_HOURS)
    return parser.parse_args(argv)


def script_epochs(dataset: str) -> int:
    """Read EPOCHS from the benchmark script of this dataset."""
    script = os.path.join(
        REPO_ROOT, "benchmarks", "examples", "great", f"run_great_{dataset}.py"
    )
    with open(script) as file:
        for line in file:
            if line.startswith("EPOCHS"):
                return int(line.split("=")[1].split("#")[0])
    raise ValueError(f"EPOCHS not found in {script}")


def main(argv=None) -> None:
    args = parse_args(argv)
    timings = measure(
        args.dataset,
        epochs=args.epochs,
        train_rows=args.train_rows,
        sample_rows=args.sample_rows,
        device=args.device,
        batch_size=args.batch_size,
        llm=args.llm,
        compare_cpu=args.compare_cpu,
    )
    planned = args.planned_epochs or script_epochs(args.dataset)
    result = estimate(
        seconds_per_epoch=timings["seconds_per_epoch_timed"],
        timed_rows=timings["timed_rows"],
        full_rows=timings["full_rows"],
        rows_per_second=timings["rows_per_second"],
        planned_epochs=planned,
        load_seconds=timings["seconds_load"],
        budget_seconds=args.budget,
    )
    report = {"timings": timings, "estimate": result}

    out_dir = os.path.join(REPO_ROOT, "benchmarks", "results", "great_timing")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"great_{args.dataset}_timing.json")
    with open(out_path, "w") as file:
        json.dump(report, file, indent=2)

    print(f"\nGReaT timing for {args.dataset} on {args.device}")
    print(f"  one epoch on all rows : {result['seconds_per_epoch_full']:.0f} s")
    print(f"  one sampling pass     : {result['seconds_per_sampling_pass']:.0f} s")
    print(f"  sampling passes       : {result['sampling_passes']}")
    print(
        f"  planned ({planned} epochs)   : {result['seconds_planned_total'] / 60:.0f} min"
    )
    print(f"  fits the budget       : {result['planned_fits_budget']}")
    print(f"  safe epochs           : {result['safe_epochs']}")
    if timings["seconds_sample_cpu_small_batch"]:
        ratio = timings["seconds_sample_cpu_small_batch"] / timings["seconds_sample"]
        print(f"  CPU small batch is    : {ratio:.1f}x slower than {args.device}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
