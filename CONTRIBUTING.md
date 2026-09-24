# Katabatic Development Guide

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

This guide documents the contribution workflow and internal architecture of Katabatic, a
framework for tabular synthetic data generation research.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Codebase Structure](#codebase-structure)
- [Development Workflow](#development-workflow)
- [Adding New Models](#adding-new-models)
- [Adding New Datasets](#adding-new-datasets)
- [Adding New Pipelines](#adding-new-pipelines)
- [Adding New Evaluations](#adding-new-evaluations)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

Katabatic follows a modular architecture with three loosely coupled components — models,
pipelines, and evaluations — each defined by an abstract base class. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the component diagrams and data flow; this section
covers the design principles relevant to contributors.

### Core Design Principles

1. **Extensibility**: each component inherits from a base class with a defined interface.
2. **Modularity**: models, pipelines, and evaluations are loosely coupled.
3. **Configurability**: pipelines support different model/evaluation combinations.
4. **Reproducibility**: seeded random state throughout training and evaluation.

### Key Design Patterns

- **Abstract base classes**: models inherit from `katabatic.models.base_model.Model`, pipelines
  from `katabatic.pipeline.base_pipeline.Pipeline`, and evaluations from
  `katabatic.evaluate.base_evaluation.Evaluation` (`TSTREvaluation` is a documented exception —
  see [ARCHITECTURE.md](ARCHITECTURE.md)).
- **Registry-based instantiation**: `ModelRegistry.load_model()`
  (`katabatic/models/registry.py`) looks up and constructs models by name, so pipelines depend
  on the `Model` interface rather than a concrete class.
- **Data flow**: raw data → preprocessing → train/test split → model training → synthetic
  generation → evaluation.

## Codebase Structure

See the [Project Structure](README.md#project-structure) section of the README for the current top-level layout — it's kept accurate there rather than duplicated here. In short: `katabatic/models/<model_name>/` holds one implementation per model (no per-model `pyproject.toml`/`poetry.lock` — dependencies live in the root `pyproject.toml` as extras, see [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md)), `katabatic/pipeline/` and `katabatic/evaluate/` hold pipelines and evaluations, and `katabatic/artifacts/` holds the versioned run-output store.

## Development Workflow

### 1. Setting Up Development Environment

See the README's [Installation](README.md#installation) section for the current setup steps (Python version, Poetry, and the extras install matrix). For contributor work you'll typically want:

```bash
poetry install --with dev -E ganblr -E ctgan -E pategan -E eval
poetry env activate
```

### 2. Development Process

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/new-model-name
   ```

2. **Implement Changes** (see specific sections below)

3. **Test Implementation**

   ```bash
   # Fast tests, no model extras (see "Testing and Quality Assurance" below
   # for why the full suite must not be run in a single process)
   make test

   # Exercise the change interactively
   jupyter lab example.ipynb
   ```

4. **Update Documentation**

   - Update this CONTRIBUTING.md if architecture changes
   - Update README.md for user-facing changes
   - Add docstrings and type hints

5. **Submit Pull Request**

## Adding New Models

Adding a model — directory layout, the `Model` base interface, registering it in `ModelRegistry`, declaring its dependency extra, and writing its integration test — is documented in full in [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md). That's the canonical reference; follow it rather than duplicating the steps here.

## Adding New Datasets

Any CSV works with the pipeline directly — `TrainTestSplitPipeline.run(input_csv=...)` needs
no registration. To add one to the shipped example catalogue instead (so it ships with the
package and is available to all users):

1. Drop the CSV in `katabatic/datasets/` and add a section documenting it (task type,
   fields, license, citation) to `katabatic/datasets/README.md`.
2. If a model should only run on certain kinds of data (e.g. numeric-only, a specific task
   type, a class-count range), declare that on the model instead of the dataset: add a
   `dataset_requirements` dict (`allowed_tasks`, `min_classes`/`max_classes`,
   `requires_numeric_only`) to that model's entry in `katabatic/models/registry.py`. See
   `katabatic/datasets/compatibility.py::check_dataset_for_model` for how it's checked.

`katabatic/datasets/registry.py`'s `DatasetRegistry` is a separate thing — it's runtime
bookkeeping the artifact pipeline uses to profile and version datasets as they're run
(`register_if_absent`), not something you edit by hand to add a catalogue entry.

## Adding New Pipelines

Create `katabatic/pipeline/your_pipeline_name/{__init__.py,pipeline.py}`. The base class is
intentionally minimal — `Pipeline.__init__(self, model)` stores the model, and `run(*args,
**kwargs)` is yours to implement (it's `NotImplementedError` in the base class):

```python
# katabatic/pipeline/your_pipeline_name/pipeline.py
from katabatic.pipeline.base_pipeline import Pipeline


class YourPipeline(Pipeline):
    """Describe the workflow this pipeline runs."""

    def run(self, *args, **kwargs):
        # e.g. preprocess -> train self.model -> evaluate -> return a results dict
        ...
```

```python
# katabatic/pipeline/your_pipeline_name/__init__.py
from .pipeline import YourPipeline

__all__ = ["YourPipeline"]
```

See `katabatic/pipeline/train_test_split/pipeline.py` for a real, fully worked implementation
(artifact-store integration, TSTR evaluation, the legacy `output_dir=` mode) to model yours on.

## Adding New Evaluations

Two conventions coexist here — pick the one matching where your evaluation needs to run.
They aren't automatically interchangeable, but a class can support both by adding a thin
`from_artifact()` adapter; see `FidelityEvaluation` below for a worked example
(`tests/test_train_test_split_pipeline.py::test_artifact_fidelity_evaluation_smoke` exercises it
end to end).

**In-memory (`SyntheticEvaluationPipeline`, `katabatic/pipeline/evaluation_pipeline.py`)** —
the `Evaluation` base class takes real and synthetic data directly as DataFrames:

```python
# katabatic/evaluate/your_evaluation_name/evaluation.py
from katabatic.evaluate.base_evaluation import Evaluation


class YourEvaluation(Evaluation):
    """Describe the evaluation methodology."""

    def evaluate(self) -> dict:
        # self.real_data / self.synthetic_data are pandas DataFrames
        ...
        return {"score": ...}
```

```python
# katabatic/evaluate/your_evaluation_name/__init__.py
from .evaluation import YourEvaluation

__all__ = ["YourEvaluation"]
```

Model yours on `katabatic/evaluate/fidelity/evaluation.py`. To actually run it, also add
the dimension name to `_AVAILABLE_DIMENSIONS` and to the dispatch in
`SyntheticEvaluationPipeline._build_evaluator()` — dimensions aren't a registry, they're
wired directly into that pipeline.

**Artifact-pipeline (`TrainTestSplitPipeline`)** — evaluations here read from directories
(`synthetic_dir`, `real_test_dir`) rather than taking DataFrames, and don't subclass
`Evaluation`. Model yours on `katabatic/evaluate/tstr/evaluation.py` (TSTR: trains
classifiers on synthetic data, scores them on real held-out data), including its
`from_artifact(store, model_ref, dataset_ref, **kwargs)` classmethod — that's what
`TrainTestSplitPipeline` actually calls when your evaluation is in its `evaluations=[...]` list.

**Supporting both** — give your DataFrame-based `Evaluation` subclass a `from_artifact()`
classmethod that reads the pipeline's `x_synth.csv`/`y_synth.csv`/`x_test.csv`/`y_test.csv`,
recombines them into `real_data`/`synthetic_data` frames, and constructs your class with them
plus `_artifact_store`/`_evaluation_ref`/`_artifact_report_relpath` kwargs so `evaluate()` can
write a report when called through the artifact pipeline (and skip that write when called
directly with DataFrames). `FidelityEvaluation.from_artifact()` is the reference implementation.

In either case, call `store.pull(path)` for each artifact path `from_artifact()` reads, so remote
stores download files written on another machine. It's a no-op for `LocalArtifactStore`.

## Testing and Quality Assurance

### Running Tests Locally

Test one model at a time. TensorFlow (`ganblr`, `pategan`) and PyTorch (`ctgan`,
`tabsyn`, `great`) segfault if both end up loaded in the same interpreter process,
so installing every model extra and running `pytest` over the whole `tests/`
directory isn't a realistic local workflow:

```text
tests/test_integration_ganblr.py + tests/test_integration_ctgan.py   -> Segmentation fault
tests/test_integration_ganblr.py + tests/test_integration_pategan.py -> 4 passed
```

```bash
make test                     # fast tests, no model extras (matches CI's lint-and-test job)
make integration MODEL=ganblr # integration tests for a single model extra
make contract                 # model promotion contract for every supported model — safe with
                               # -E all installed, since each model's case runs pytest-forked
```

Run `make help` for the full target list, and see [.github/workflows/ci.yml](.github/workflows/ci.yml)
for how these map onto CI jobs: `lint-and-test`, and `integration-<model>` (which also runs the
model promotion contract as a second step, per model).

### Direct pytest usage

```bash
poetry install --with dev

# Fast tests (default CI, excludes the model promotion contract)
poetry run pytest -q --ignore=tests/test_model_registry.py

# Integration tests for one model (install its extra first)
poetry install --with dev -E ganblr
poetry run pytest -m "integration and ganblr"
```

Currently **supported** models are whatever `ModelRegistry.get_supported_models()` returns —
see [docs/EXPERIMENTAL_MODELS.md](docs/EXPERIMENTAL_MODELS.md). New models remain experimental
until they have an extra, registry entry, and integration coverage; see
[MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md).

### Writing Tests

Tests live flat under `tests/` (there's no `unit/`/`integration/` split): shared fixtures in
`tests/conftest.py` (e.g. `tiny_binary_csv`), component tests such as
`tests/test_train_test_split_pipeline.py` and `tests/test_datasets.py`, and one
`tests/test_integration_<model_name>.py` per model, guarded with `pytest.importorskip(...)` for
its heavy dependencies and marked `@pytest.mark.integration` plus a model-specific marker
(`ganblr`, `ctgan`, `pategan`, ...; markers are declared under `[tool.pytest.ini_options]` in
`pyproject.toml`).

For a new model, follow the pattern in `tests/test_integration_ganblr.py` /
`tests/test_integration_ctgan.py`: run it through `TrainTestSplitPipeline` with a
`LocalArtifactStore` and assert the expected model/synthetic/evaluation artifacts are written.
See [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md) for the full checklist (test harness,
promotion contract, CI wiring). For a new pipeline or evaluation, test initialization, the
`run`/`evaluate` method under representative inputs, and error handling for invalid inputs.

### Code Quality Checks

```bash
make format    # ruff format + ruff check --fix + pre-commit run --all-files
make security  # bandit security scan

# Or individually
poetry run ruff format katabatic tests
poetry run ruff check --no-cache katabatic tests
poetry run bandit -r katabatic -ll
poetry run mypy katabatic   # optional, not yet enforced in CI
```

CI enforces coverage in three layers (see [.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **`lint-and-test`** enforces a 70% floor scoped to (`pipeline/`, `utils/`, `datasets/`,
`artifacts/`, `evaluate/`, `registry.py`, `base_model.py`).
- **`integration`** reports each model's own coverage (`katabatic/models/<model>/*`) without
gating it. This is to avoid breaking builds as they come in.
- **`merge-coverage`** enforces a 45% floor across the *whole* codebase, but only when the full
  model matrix has run. A PR that only touches one model's files gets a combined coverage report,
  just without a gate.

There's no separate per-PR minimum for new code beyond the above.

## Usage Examples

See the README's [Quick Start](README.md#quick-start) for the artifact-pipeline example, and
[GANBLR_FLOW.md](GANBLR_FLOW.md) for a full walkthrough (preprocess → train → sample → evaluate,
versioned under `artifacts/` via `LocalArtifactStore`).

## Best Practices

- **Type hints & docstrings**: add type hints to public methods; NumPy-style docstrings for classes/methods.
- **Single responsibility**: keep models, pipelines, and evaluations loosely coupled — inject the model into the pipeline rather than hard-coding it.
- **Reproducibility**: use fixed random seeds in tests and examples.
- **Performance**: prefer vectorized NumPy/Pandas operations for large datasets; be mindful of memory with in-memory synthetic data generation.

## Troubleshooting

**Import errors** — models are nested one level deeper than the package name suggests:

```python
from katabatic.models.ganblr.models import GANBLR   # correct
from katabatic.models.ganblr import GANBLR           # wrong
```

**Missing dependencies** — install the model's extra (dependencies live in the root
`pyproject.toml`, not per-model):

```bash
poetry install -E your_model_name   # or: pip install katabatic[your_model_name]
```

**Data format** — `y` should be a 1D array, `X` a DataFrame:

```python
y = pd.read_csv("y_train.csv").values.ravel()
X = pd.read_csv("x_train.csv")
```

For anything else: read the full stack trace, compare against `example.ipynb` or the
`tests/test_integration_*.py` files for a working reference, and isolate the issue with a
minimal reproduction before opening an issue.
