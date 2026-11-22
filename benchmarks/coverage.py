"""Report benchmark coverage and rank models by their composite scores."""

import json
import math
from pathlib import Path

DATASETS = ["adult", "shuttle", "car", "magic", "nursery"]

BENCHMARKS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BENCHMARKS_DIR / "examples"
RESULTS_DIR = BENCHMARKS_DIR / "results"

SUCCESS = "Success"
FAILED = "Failed"
UNSUPPORTED = "Unsupported"
UNTESTED = "Untested"


def get_models(examples_dir: Path = EXAMPLES_DIR) -> list[str]:
    """Return model names from the benchmark example directories."""
    if not examples_dir.exists():
        return []

    return sorted(path.name for path in examples_dir.iterdir() if path.is_dir())


def has_benchmark_script(
    model: str,
    dataset: str,
    examples_dir: Path = EXAMPLES_DIR,
) -> bool:
    """Check whether a benchmark script exists for a model and dataset."""
    model_dir = examples_dir / model
    return any(model_dir.glob(f"run_*_{dataset}.py"))


def read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it is missing or invalid."""
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    return data if isinstance(data, dict) else None


def get_composite_score(report_path: Path) -> float | None:
    """Return a valid composite score from an evaluation report."""
    report = read_json(report_path)
    if report is None:
        return None

    score = report.get("composite_score")

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None

    if not math.isfinite(score):
        return None

    return float(score)


def get_cell_status(
    model: str,
    dataset: str,
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """Determine the coverage status for one model and dataset."""
    if not has_benchmark_script(model, dataset, examples_dir):
        return UNSUPPORTED

    result_dir = results_dir / dataset / model
    report_path = result_dir / f"{model}_{dataset}_evaluation_report.json"
    runtime_path = result_dir / f"{model}_{dataset}_runtime.json"

    runtime = read_json(runtime_path) if runtime_path.exists() else None

    if runtime_path.exists():
        if runtime is None:
            return FAILED

        if runtime.get("status") in ("failed", "incomplete"):
            return FAILED

    if not report_path.exists():
        if runtime is not None and runtime.get("status") == "completed":
            return FAILED
        return UNTESTED

    if get_composite_score(report_path) is None:
        return FAILED

    return SUCCESS


def build_coverage_table(
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Build a model-by-dataset coverage table."""
    datasets = datasets or DATASETS
    models = get_models(examples_dir)

    return {
        model: {
            dataset: get_cell_status(model, dataset, examples_dir, results_dir)
            for dataset in datasets
        }
        for model in models
    }


def build_leaderboard(
    examples_dir: Path = EXAMPLES_DIR,
    results_dir: Path = RESULTS_DIR,
    datasets: list[str] | None = None,
) -> list[dict]:
    """Rank models by their average successful composite score."""
    datasets = datasets or DATASETS
    leaderboard = []

    for model in get_models(examples_dir):
        scores = []

        for dataset in datasets:
            if get_cell_status(model, dataset, examples_dir, results_dir) != SUCCESS:
                continue

            report_path = (
                results_dir
                / dataset
                / model
                / f"{model}_{dataset}_evaluation_report.json"
            )
            score = get_composite_score(report_path)

            if score is not None:
                scores.append((dataset, score))

        average = sum(score for _, score in scores) / len(scores) if scores else None

        leaderboard.append(
            {
                "model": model,
                "average_score": average,
                "successful_runs": len(scores),
                "datasets": [dataset for dataset, _ in scores],
            }
        )

    return sorted(
        leaderboard,
        key=lambda item: (
            item["average_score"] is None,
            -(item["average_score"] or 0),
            item["model"],
        ),
    )


def main() -> None:
    """Print the coverage table and model leaderboard."""
    coverage = build_coverage_table()
    datasets = DATASETS

    print("BENCHMARK COVERAGE")
    print(f"{'Model':<22}" + "".join(f"{d:<14}" for d in datasets))

    for model, statuses in coverage.items():
        print(f"{model:<22}" + "".join(f"{statuses[d]:<14}" for d in datasets))

    total_cells = len(coverage) * len(datasets)
    counts = {
        status: sum(
            cell == status
            for statuses in coverage.values()
            for cell in statuses.values()
        )
        for status in (SUCCESS, FAILED, UNSUPPORTED, UNTESTED)
    }

    print(f"\nTotal cells: {total_cells}")
    for status, count in counts.items():
        print(f"{status}: {count}")

    print("\nMODEL LEADERBOARD")
    print(f"{'Rank':<8}{'Model':<22}{'Average score':<18}{'Successful runs'}")

    for rank, entry in enumerate(build_leaderboard(), start=1):
        score = (
            f"{entry['average_score']:.4f}"
            if entry["average_score"] is not None
            else "N/A"
        )
        print(f"{rank:<8}{entry['model']:<22}{score:<18}{entry['successful_runs']}")


if __name__ == "__main__":
    main()
