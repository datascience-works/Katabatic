# Katabatic

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

A framework for synthetic tabular data generation: 16 supported generative models behind one interface, a six-dimension evaluation, and versioned artifacts that can live on local disk or in cloud storage.

## Features

- **Supported Generative Models**: GAN-based (GANBLR, CTGAN, PATE-GAN, TVAE-GAN), diffusion (TabSyn, FairTabDiffusion), language-model (GReaT, REaLTabFormer), differentially private (MST, PrivTree, PATE-GAN) and statistical baselines (ARF, KDE, Histogram, NaiveBayes, SMOTE, SynthPop). Experimental models live in `katabatic.experimental`; see [docs/EXPERIMENTAL_MODELS.md](https://github.com/datascience-works/Katabatic/blob/main/docs/EXPERIMENTAL_MODELS.md)
- **Automated Pipeline**: End-to-end training, generation, and evaluation workflows
- **Six-Dimension Evaluation**: `model.evaluate()` scores fidelity, utility (TSTR), diversity, privacy, consistency and stability, plus a weighted composite
- **Versioned Artifacts**: datasets, trained models and evaluations stored locally or in S3, GCS or Azure
- **Data Preprocessing**: `preprocess_tabular()` cleans raw CSVs before training (missing values, empty and constant columns, target last)
- **Extensible Architecture**: Bring your own model (subclass `Model`) or evaluation (`evaluate(pipeline=...)`)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Usage](#usage)
- [Models](#models)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11.x (strictly required due to TensorFlow compatibility) — via [pyenv](https://github.com/pyenv/pyenv) (macOS/Linux) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (Windows), or any other installer
- **[Poetry](https://python-poetry.org/docs/#installation)**: `curl -sSL https://install.python-poetry.org | python3 -`
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional, for GReaT model training)

## Installation

```bash
git clone https://github.com/datascience-works/Katabatic.git
cd Katabatic
pyenv local 3.11.9   # or otherwise select Python 3.11.x
```

Once Python 3.11 and Poetry are on `PATH`, `scripts/setup.py` verifies both and runs the install
for you (cross-platform, checked in CI on Linux/macOS/Windows):

```bash
python scripts/setup.py                # core install
python scripts/setup.py --model ganblr # core + one model extra
```

Or install directly with Poetry / pip — useful for installing several extras at once:

**Install matrix (PyPI / Poetry extras):**

| Use case | Command |
| ---------- | --------- |
| Core only | `pip install katabatic` or `poetry install` |
| GANBLR (supported) | `pip install "katabatic[ganblr]"` or `poetry install -E ganblr` |
| CTGAN (supported) | `pip install "katabatic[ctgan]"` or `poetry install -E ctgan` |
| PATE-GAN (supported) | `pip install "katabatic[pategan]"` or `poetry install -E pategan` |
| TabSyn (supported) | `pip install "katabatic[tabsyn]"` or `poetry install -E tabsyn` |
| GReaT (supported) | `pip install "katabatic[great]"` or `poetry install -E great` |
| SMOTE (supported) | `pip install "katabatic[smote]"` or `poetry install -E smote` |
| MST (supported) | `pip install "katabatic[mst]" "private-pgm @ git+https://github.com/ryan112358/private-pgm.git@01f02f17eba440f4e76c1d06fa5ee9eed0bd2bca"` or `poetry install -E mst --with mst`: private-pgm isn't on PyPI, so it installs separately |
| PrivTree (supported) | `pip install "katabatic[privtree]"` or `poetry install -E privtree` |
| ARF (supported) | `pip install "katabatic[arf]"` or `poetry install -E arf` |
| SynthPop (supported, requires R) | `pip install "katabatic[synthpop]"` or `poetry install -E synthpop` — also needs R + CRAN `synthpop` packages, see [katabatic/models/synthpop/README.md](https://github.com/datascience-works/Katabatic/blob/main/katabatic/models/synthpop/README.md) |
| NaiveBayes (supported) | `pip install "katabatic[naivebayes]"` or `poetry install -E naivebayes` |
| Histogram (supported) | `pip install "katabatic[histogram]"` or `poetry install -E histogram` |
| REaLTabFormer (supported) | `pip install "katabatic[realtabformer]"` or `poetry install -E realtabformer` |
| KDE (supported) | `pip install "katabatic[kde]"` or `poetry install -E kde` |
| FairTabDiffusion (supported) | `pip install "katabatic[fairtabdiffusion]"` or `poetry install -E fairtabdiffusion` |
| TVAE-GAN (supported) | `pip install "katabatic[tvaegan]"` or `poetry install -E tvaegan` |
| TSTR + XGBoost | `pip install "katabatic[eval]"` or `poetry install -E eval` |
| Several models | `pip install "katabatic[ganblr,ctgan]"` or `poetry install -E ganblr -E ctgan` |
| Development | `poetry install --with dev` |

Experimental models (TabEBM, TabDDPM, GANBLR++, CoDi, MedGAN, GMM, TabKDE) live in `katabatic.experimental.models`, with no API stability guarantee; see [docs/EXPERIMENTAL_MODELS.md](https://github.com/datascience-works/Katabatic/blob/main/docs/EXPERIMENTAL_MODELS.md).
For contributor work: `poetry install --with dev -E ganblr -E ctgan -E pategan -E eval && poetry env activate`.

For GPU training, install the PyTorch build for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) into the same environment.

### Verify Installation

```bash
python -c "import katabatic; print(katabatic.__version__)"
python -c "from katabatic.models.registry import ModelRegistry; print(ModelRegistry.get_supported_models())"
```

## Quick Start

### Artifact pipeline (recommended)

Versioned datasets, models, and evaluations under `artifacts/`. See [GANBLR_FLOW.md](https://github.com/datascience-works/Katabatic/blob/main/GANBLR_FLOW.md) for details.

```python
from importlib.resources import files

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline import TrainTestSplitPipeline
from katabatic.utils.preprocess import preprocess_tabular

# The car dataset ships with the package.
preprocess_tabular(str(files("katabatic.datasets") / "car.csv"), "preprocessed_data/car.csv")

store = LocalArtifactStore("artifacts")
pipeline = TrainTestSplitPipeline(model=GANBLR())
results = pipeline.run(
    input_csv="preprocessed_data/car.csv",
    dataset_name="car",
    artifact_store=store,
    model_name="ganblr",
)
# results["model_ref"], results["evaluation_refs"] — TSTR metrics on disk
```

#### Remote artifact stores

To keep artifacts in S3, GCS or Azure, so a model trained on one machine can be reloaded or
evaluated on another, pass an `FsspecArtifactStore` as `artifact_store=` instead:

```bash
pip install "katabatic[artifacts-s3]"  # or artifacts-gcs, artifacts-azure; combine for several clouds
```

```python
from katabatic.artifacts import FsspecArtifactStore

store = FsspecArtifactStore("s3://my-bucket/katabatic", local_cache_dir="artifact-cache")
```

> **Experimental:** `FsspecArtifactStore` is tested against fsspec's in-memory filesystem but
> not yet against real S3, GCS or Azure buckets
> ([#239](https://github.com/datascience-works/Katabatic/issues/239)). Until it is, its API may
> change in a minor release: it's the one exception to semantic versioning outside
> `katabatic.experimental`.

Files are cached locally and transferred only when they change; the pipeline and
`load_from_ref()` handle this automatically. A store never overwrites a remote file it hasn't
read at its current version: `save_json()`, `save_bytes()` and `sync()` raise
`ArtifactConflictError` instead, so load the file, re-apply your change and save again.
`exists()` downloads the file it checks. The store is thread-safe, but don't share one
`local_cache_dir` between processes.

**Only load models from storage you trust.** Most models save their state with pickle, so
`load_from_ref()` runs code from the store.

## Examples

The notebooks in [examples/](https://github.com/datascience-works/Katabatic/tree/main/examples) run with the core install and are executed in CI:

| Notebook | Shows |
| --- | --- |
| [quickstart.ipynb](https://github.com/datascience-works/Katabatic/blob/main/examples/quickstart.ipynb) | Pick any model by name, train it through the pipeline, generate rows and reload the trained model |
| [evaluation.ipynb](https://github.com/datascience-works/Katabatic/blob/main/examples/evaluation.ipynb) | Score a model on the six dimensions, run a subset, or plug in your own evaluation |
| [remote_artifact_store.ipynb](https://github.com/datascience-works/Katabatic/blob/main/examples/remote_artifact_store.ipynb) | Share artifacts between machines through S3, GCS or Azure |

Open them in VS Code, or install Jupyter first (`pip install jupyterlab`). Per-model benchmark
scripts are in [benchmarks/examples/](https://github.com/datascience-works/Katabatic/tree/main/benchmarks/examples).

## Usage

### Data Preprocessing

`preprocess_tabular()` cleans a raw CSV before training: `?` is treated as missing, empty and
constant columns are dropped, missing numbers are filled with the column median and missing
text with `"Missing"`, and the target moves to the last column. It doesn't encode or discretise;
each model handles its own data representation.

```python
from katabatic.utils.preprocess import preprocess_tabular

preprocess_tabular(
    file_path="raw_data/your_dataset.csv",
    output_path="preprocessed_data/your_dataset.csv",
    target_col="income",  # optional: defaults to the last column
)
```

### Training Models

#### GANBLR Model

```python
from katabatic.models.ganblr.models import GANBLR
import pandas as pd

# Load your data
X = pd.read_csv("path/to/features.csv")
y = pd.read_csv("path/to/labels.csv").values.ravel()

# Initialize and train model
model = GANBLR()
model.fit(X, y, k=2, epochs=100, batch_size=64)

# Generate synthetic data
synthetic_data = model.sample(n_samples=1000)
```

#### GReaT Model

```python
from katabatic.models.great.models import GReaT
import pandas as pd

# Load your data
data = pd.read_csv("path/to/your_data.csv")

# Initialize and train model
model = GReaT(
    llm='gpt2',  # any Hugging Face causal language model
    epochs=100,
    batch_size=8
)

model.fit(data)

# Generate synthetic data
synthetic_data = model.sample(
    n_samples=1000,
    temperature=0.7
)
```

### Pipeline Usage

The artifact-store flow shown in [Quick Start](#quick-start) is the recommended way to run
`TrainTestSplitPipeline`. It also supports a legacy, non-artifact-store mode — pass `output_dir=`
instead of `artifact_store=`/`dataset_name=` — see [GANBLR_FLOW.md](https://github.com/datascience-works/Katabatic/blob/main/GANBLR_FLOW.md#legacy-directory-layout-optional)
for that layout.

## Models

Most supported models have a README in `katabatic/models/<name>/` covering the paper, parameters
and benchmark results; [docs/EXPERIMENTAL_MODELS.md](https://github.com/datascience-works/Katabatic/blob/main/docs/EXPERIMENTAL_MODELS.md) lists them all. A few of
the most used:

### GANBLR (GAN-based Bayesian Learning Rules)

- **Type**: GAN-based generative model
- **Best for**: Discrete/categorical tabular data
- **Features**:
  - k-dependence Bayesian Networks
  - Adversarial training
  - High-quality discrete data generation

### CTGAN (Conditional Tabular GAN)

- **Type**: GAN-based generative model
- **Best for**: Mixed-type tabular data with imbalanced categorical columns
- **Features**:
  - Conditional generator over categorical columns
  - Mode-specific normalization for continuous columns
  - WGAN-GP training (Torch backend), with a NumPy fallback

### PATE-GAN (Private Aggregation of Teacher Ensembles GAN)

- **Type**: Differentially private GAN-based generative model
- **Best for**: Tabular data requiring formal privacy guarantees
- **Features**:
  - WGAN-GP adversarial training
  - Differential privacy via a Gaussian noise mechanism
  - Adapted from Jordon et al. (ICLR 2019) — see [katabatic/models/pategan/README.md](https://github.com/datascience-works/Katabatic/blob/main/katabatic/models/pategan/README.md) for how this implementation differs from the paper

### TabSyn (Score-based Diffusion in Latent Space)

- **Type**: Diffusion-based generative model
- **Best for**: Mixed numerical and categorical tabular data
- **Features**:
  - Transformer-based VAE encoder/decoder
  - Diffusion model trained on the learned latent space
  - Based on Zhang et al. (ICLR 2024)

### GReaT (Generation of Realistic Tabular Data)

- **Type**: Transformer-based generative model
- **Best for**: Mixed data types (numerical + categorical)
- **Features**:
  - Pre-trained language model fine-tuning
  - Conditional generation
  - Data imputation capabilities

## Datasets

Models are benchmarked against five datasets in the data catalogue — see
[katabatic/datasets/README.md](https://github.com/datascience-works/Katabatic/blob/main/katabatic/datasets/README.md) for details on each.

## Evaluation

### Scoring a trained model

Every model inherits `evaluate()`, which scores it on six dimensions (fidelity, utility, diversity,
privacy, consistency and stability) plus a weighted composite, the same scoring the benchmark
scripts use:

```python
report = model.evaluate(train_df, target_col="class", test_data=test_df)
report.dimension_scores   # {"fidelity": 0.98, "utility": 0.91, ...}
report.composite_score    # 0.87
```

Pass `dimensions=["utility"]` to run a subset, or `synthetic_data=...` to score data you already
generated. If a dimension fails, `evaluate()` warns, leaves it out of the composite and records
the error in `report.errors`; the other dimensions still run.

To use your own evaluation, pass it as `pipeline=`, in the same way scikit-learn takes a `scoring=`
argument. It can be any object with a `run()` method. `evaluate()` still does the sampling and
column alignment, and passes the model along; accept `**kwargs` for the arguments you don't need:

```python
class MyEvaluation:
    def run(self, real_data, synthetic_data, target_col=None, **kwargs):
        return {"mean_gap": (real_data.mean(numeric_only=True)
                             - synthetic_data.mean(numeric_only=True)).abs().mean()}

model.evaluate(train_df, pipeline=MyEvaluation())
```

See `Model.evaluate()` and `EvaluationPipeline` in `katabatic/models/base_model.py` for all options.

### TSTR (Train on Synthetic, Test on Real)

Katabatic includes comprehensive evaluation using the TSTR methodology:

```python
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

# Initialize evaluator
evaluator = TSTREvaluation(
    synthetic_dir="path/to/synthetic/data",
    real_test_dir="path/to/real/test/data"
)

# Run evaluation with multiple ML models
results = evaluator.evaluate()
```

**Supported Evaluation Models:**

- Logistic Regression
- Multi-layer Perceptron (MLP)
- Random Forest
- XGBoost

**Metrics:**

- Accuracy
- F1 Score
- AUC-ROC (for binary classification)

**Statistical fidelity** (categorical JSD, continuous Wasserstein distance, correlation
preservation, and DCR — distance to closest record) is available via
`katabatic.evaluate.fidelity.evaluation.FidelityEvaluation` in both `SyntheticEvaluationPipeline`
(DataFrame-based) and, via its `from_artifact()` adapter, `TrainTestSplitPipeline`
(artifact-store-based) runs.

## Development

### Recommended VS Code Extensions

`ms-python.python`, `charliermarsh.ruff`, `ms-toolsai.jupyter`

### Development Setup

From the cloned repo root (see [Installation](#installation)):

```bash
poetry install --with dev -E ganblr -E eval   # add -E {model} as needed

poetry check
poetry run ruff check --no-cache katabatic tests
poetry run pytest                              # fast unit tests
poetry run pytest -m integration               # after installing model extras
poetry run mypy katabatic/                     # optional
```

### Project Structure

```text
Katabatic/
├── katabatic/                 # Installable package (PyPI wheel)
│   ├── models/                # supported models, base Model class, ModelRegistry
│   ├── experimental/          # experimental models (no API stability guarantee)
│   ├── pipeline/              # TrainTestSplitPipeline, SyntheticEvaluationPipeline
│   ├── evaluate/              # TSTR and the six evaluation dimensions
│   ├── artifacts/             # Versioned store helpers
│   ├── datasets/              # packaged benchmark datasets and DatasetRegistry
│   └── utils/                 # preprocess, split_dataset, ...
├── artifacts/                 # Local run outputs (gitignored)
├── docs/                      # EXPERIMENTAL_MODELS.md
├── benchmarks/                # per-model benchmark scripts (benchmarks/examples/)
├── examples/                  # quickstart, evaluation and remote-store notebooks
├── tests/                     # Unit + integration tests
├── GANBLR_FLOW.md             # Artifact pipeline walkthrough
├── pyproject.toml
└── README.md
```

### Building from Source

```bash
# Build package
poetry build

# Install locally
pip install dist/katabatic-*.whl
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/datascience-works/Katabatic/blob/main/CONTRIBUTING.md) for the development guide
(architecture, adding pipelines/evaluations, testing) and [MODEL_CONTRIBUTIONS.md](https://github.com/datascience-works/Katabatic/blob/main/MODEL_CONTRIBUTIONS.md)
for adding or promoting a model. In short:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`, following
   [Conventional Commits](https://www.conventionalcommits.org/) — enforced by pre-commit)
4. **Push** to the branch and **open** a Pull Request into `development`

Formatting, linting, and security scans run via `pre-commit` (`ruff format`, `ruff check`, plus
the hooks in [.pre-commit-config.yaml](https://github.com/datascience-works/Katabatic/blob/main/.pre-commit-config.yaml)):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/datascience-works/Katabatic/blob/main/LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/datascience-works/Katabatic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/datascience-works/Katabatic/discussions)

## Related Projects

- [GANBLR: A Tabular Data Generation Model (ICDM 2021)](https://ieeexplore.ieee.org/document/9679177)
- [GReaT Repository](https://github.com/kathrinse/be_great)
