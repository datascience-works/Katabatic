# Katabatic Development Guide

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

This guide provides comprehensive documentation for internal development teams working on the Katabatic framework for synthetic tabular data generation.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Codebase Structure](#codebase-structure)
- [Development Workflow](#development-workflow)
- [Adding New Models](#adding-new-models)
- [Adding New Pipelines](#adding-new-pipelines)
- [Adding New Evaluations](#adding-new-evaluations)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

Katabatic follows a modular architecture with three main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Models      │    │    Pipelines    │    │   Evaluations   │
│                 │    │                 │    │                 │
│ • GANBLR        │    │ • TrainTestSplit│    │ • TSTR          │
│ • GReaT         │ ───► • CrossValidation│ ───► • Custom Evals │
│ • CustomModel   │    │ • CustomPipeline│    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Design Principles

1. **Extensibility**: Each component inherits from a base class with defined interfaces
2. **Modularity**: Models, pipelines, and evaluations are loosely coupled
3. **Configurability**: Pipeline configurations support different model-evaluation combinations
4. **Reproducibility**: Built-in support for seeds and experiment tracking

## 📁 Codebase Structure

See the [Project Structure](README.md#project-structure) section of the README for the current top-level layout — it's kept accurate there rather than duplicated here. In short: `katabatic/models/<model_name>/` holds one implementation per model (no per-model `pyproject.toml`/`poetry.lock` — dependencies live in the root `pyproject.toml` as extras, see [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md)), `katabatic/pipeline/` and `katabatic/evaluate/` hold pipelines and evaluations, and `katabatic/artifacts/` holds the versioned run-output store.

### Key Design Patterns

#### 1. Abstract Base Classes

- **Models**: All models inherit from `katabatic.models.base_model.Model`
- **Pipelines**: All pipelines inherit from `katabatic.pipeline.base_pipeline.Pipeline`
- **Evaluations**: All evaluations inherit from `katabatic.evaluate.base_evaluation.Evaluation`

#### 2. Factory Pattern

- Pipelines instantiate models dynamically
- Evaluations are configurable and pluggable

#### 3. Data Flow

```
Raw Data → Preprocessing → Train/Test Split → Model Training → Synthetic Generation → Evaluation → Results
```

## 🔄 Development Workflow

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
   # Run existing tests
   pytest tests/

   # Test with example notebook
   jupyter lab example.ipynb
   ```

4. **Update Documentation**

   - Update this CONTRIBUTING.md if architecture changes
   - Update README.md for user-facing changes
   - Add docstrings and type hints

5. **Submit Pull Request**

## 🤖 Adding New Models

Adding a model — directory layout, the `Model` base interface, registering it in `ModelRegistry`, declaring its dependency extra, and writing its integration test — is documented in full in [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md). That's the canonical reference; follow it rather than duplicating the steps here.

## 🔄 Adding New Pipelines

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

## 📊 Adding New Evaluations

Create `katabatic/evaluate/your_evaluation_name/{__init__.py,evaluation.py}`. The base class
takes real and synthetic data directly as DataFrames and `evaluate()` returns a result dict:

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

See `katabatic/evaluate/tstr/evaluation.py` (TSTR: trains classifiers on synthetic data, scores
them on real held-out data) for a real, fully worked implementation to model yours on.

## 🧪 Testing and Quality Assurance

### Running the full suite locally

Do **not** run `pytest` over the whole `tests/` directory with more than one
model backend installed. TensorFlow (`ganblr`, `pategan`) and PyTorch (`ctgan`)
segfault when run on the same interpreter process:

```
tests/test_integration_ganblr.py + tests/test_integration_ctgan.py   -> Segmentation fault
tests/test_integration_ganblr.py + tests/test_integration_pategan.py -> 4 passed
```

Every file passes on its own. Use one of the `Makefile` targets, which run one pytest
process per file/model, or install a single extra at a time:

```bash
make test                     # fast tests, no model extras (matches CI's lint-and-test job)
make test-all                 # every tests/test_*.py, one pytest process each
make integration MODEL=ganblr # integration tests for a single model extra
make contract                 # model promotion contract (tests/test_model_registry.py), installs -E all
```

Run `make help` for the full target list, and see [.github/workflows/ci.yml](.github/workflows/ci.yml)
for how these map onto CI jobs (`lint-and-test`, `integration`, `model-contract`).

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
promotion contract, CI wiring).

### Test Requirements for New Components

**For New Models:**

1. Test inheritance from base `Model` class
2. Test initialization and parameter handling
3. Test `train`/`evaluate`/`sample` method interfaces
4. Test integration with the artifact pipeline (train → sample → evaluate → reload)
5. Test error handling and edge cases

**For New Pipelines/Evaluations:**

1. Test initialization with a model (pipelines) or with real/synthetic data (evaluations)
2. Test the `run`/`evaluate` method with representative inputs
3. Test integration with the rest of the pipeline
4. Test error conditions and edge cases

### Code Quality Checks

```bash
make format    # ruff format + ruff check --fix + pre-commit run --all-files
make security  # bandit security scan

# Or individually
poetry run ruff format katabatic tests
poetry run ruff check katabatic tests
poetry run bandit -r katabatic -ll
poetry run mypy katabatic   # optional, not yet enforced in CI
```

CI's `lint-and-test` job runs the fast suite with coverage, and `merge-coverage` fails the build
if combined coverage drops below 45% (see `[tool.coverage.run]` in `pyproject.toml` and
[.github/workflows/ci.yml](.github/workflows/ci.yml)) — there's no separate per-PR minimum for
new code.

## 📖 Usage Examples

See the README's [Quick Start](README.md#quick-start) for the current artifact-pipeline
example, and [GANBLR_FLOW.md](GANBLR_FLOW.md) for a full walkthrough (preprocess → train →
sample → evaluate, with everything versioned under `artifacts/` via `LocalArtifactStore`).
Kept here once, in one accurate place, instead of duplicated across docs.

## 🏆 Best Practices

- **Type hints & docstrings**: add type hints to public methods; NumPy-style docstrings for classes/methods.
- **Single responsibility**: keep models, pipelines, and evaluations loosely coupled — inject the model into the pipeline rather than hard-coding it.
- **Reproducibility**: use fixed random seeds in tests and examples.
- **Performance**: prefer vectorized NumPy/Pandas operations for large datasets; be mindful of memory with in-memory synthetic data generation.

## 🔧 Troubleshooting

**Import errors** — models live one level deeper than you'd expect:

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
