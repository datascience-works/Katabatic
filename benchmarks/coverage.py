"""Print a model x dataset grid of which benchmark scripts exist."""

from pathlib import Path

DATASETS = ["adult", "shuttle", "car", "magic", "nursery"]
root = Path("benchmarks/examples")
models = sorted(p.name for p in root.iterdir() if p.is_dir())
print(f"{'model':<16}" + "".join(f"{d:<9}" for d in DATASETS) + "missing")
total_missing = 0
for m in models:
    cells, missing = [], 0
    for d in DATASETS:
        ok = any((root / m).glob(f"run_*_{d}.py"))
        cells.append("yes" if ok else "NO")
        missing += not ok
    total_missing += missing
    print(f"{m:<16}" + "".join(f"{c:<9}" for c in cells) + str(missing))
print(
    f"\n{len(models)} models, {len(models) * len(DATASETS)} cells, {total_missing} missing"
)
