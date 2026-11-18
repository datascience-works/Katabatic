"""Check every benchmark script against the protocol in benchmarks/protocol.py.

Usage (from the repository root):

    python benchmarks/check_protocol.py              # summary per model
    python benchmarks/check_protocol.py --verbose    # one line per script
    python benchmarks/check_protocol.py --markdown   # table for protocol.md
    python benchmarks/check_protocol.py --strict     # exit 1 on any DIFF/MISSING

The check only reads the scripts. It never trains a model.

Statuses: OK matches the protocol, DIFF differs from it, MISSING means the
value is not set anywhere, REVIEW means a person must look, INFO is shown for
model-specific or fixed budgets, and N/A means there is nothing to check.
"""

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent / "examples"


def _constants(tree: ast.Module) -> dict:
    """Module-level NAME = <number> assignments."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _find_value(tree: ast.Module, names: tuple):
    """Return (found, value) for the first keyword or constant called one of names."""
    constants = _constants(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in names:
                    if isinstance(kw.value, ast.Constant):
                        return True, kw.value.value
                    if isinstance(kw.value, ast.Name) and kw.value.id in constants:
                        return True, constants[kw.value.id]
                    return True, "non-constant"
    for name in names:
        if name in constants:
            return True, constants[name]
    return False, None


def check_script(path: Path, model: str, dataset: str) -> dict:
    """Return the seed and budget checks for one script."""
    spec = protocol.MODEL_SPECS[model]
    tree = ast.parse(path.read_text())

    # Seed
    if spec.seed_params:
        found, seed = _find_value(tree, spec.seed_params)
        if not found:
            seed = spec.seed_default
        if seed == "non-constant":
            seed_status = "REVIEW"
        elif seed is None:
            seed_status = "MISSING"
        else:
            seed_status = "OK" if seed == protocol.SEED else "DIFF"
    else:
        seed, seed_status = None, ("N/A" if spec.policy == protocol.FIXED else "REVIEW")

    # Budget
    if spec.budget_params:
        found, budget = _find_value(tree, spec.budget_params)
        if not found:
            budget = spec.budget_default
        expected = None
        if spec.policy == protocol.STANDARD:
            expected = protocol.epochs_for(dataset)
            if budget is None:
                budget_status = "MISSING"
            else:
                budget_status = "OK" if budget == expected else "DIFF"
        else:
            budget_status = "INFO"
    else:
        budget, expected, budget_status = None, None, "N/A"

    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "seed_status": seed_status,
        "budget": budget,
        "expected": expected,
        "budget_status": budget_status,
    }


def scan(examples: Path = EXAMPLES) -> list:
    """Check every run_<model>_<dataset>.py under the examples folder."""
    rows = []
    for path in sorted(examples.glob("*/run_*_*.py")):
        model = path.parent.name
        dataset = path.stem.rsplit("_", 1)[1]
        if model in protocol.MODEL_SPECS and dataset in protocol.DATASET_SIZE:
            rows.append(check_script(path, model, dataset))
    return rows


def _summary(rows: list) -> dict:
    out = {}
    for row in rows:
        entry = out.setdefault(row["model"], {"seed": Counter(), "budget": Counter()})
        entry["seed"][row["seed_status"]] += 1
        entry["budget"][row["budget_status"]] += 1
    return out


def _fmt(counter: Counter) -> str:
    return ", ".join(f"{n} {s}" for s, n in sorted(counter.items()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    rows = scan()
    summary = _summary(rows)

    if args.markdown:
        print("| Model | Policy | Seed | Budget | Note |")
        print("| --- | --- | --- | --- | --- |")
        for model, entry in summary.items():
            spec = protocol.MODEL_SPECS[model]
            print(
                f"| {model} | {spec.policy} | {_fmt(entry['seed'])} | "
                f"{_fmt(entry['budget'])} | {spec.note} |"
            )
    elif args.verbose:
        for r in rows:
            print(
                f"{r['model']:<17}{r['dataset']:<9}"
                f"seed={r['seed']!s:<13}{r['seed_status']:<8}"
                f"budget={r['budget']!s:<8}expected={r['expected']!s:<6}{r['budget_status']}"
            )
    else:
        print(f"{'model':<18}{'policy':<10}{'seed':<24}budget")
        for model, entry in summary.items():
            policy = protocol.MODEL_SPECS[model].policy
            print(
                f"{model:<18}{policy:<10}{_fmt(entry['seed']):<24}{_fmt(entry['budget'])}"
            )

    bad = sum(
        1
        for r in rows
        for k in ("seed_status", "budget_status")
        if r[k] in ("DIFF", "MISSING")
    )
    print(f"\n{len(rows)} scripts checked, {bad} DIFF or MISSING findings")
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
