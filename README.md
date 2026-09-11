# Katabatic

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

A comprehensive framework for synthetic tabular data generation using state-of-the-art machine learning models including GANBLR, CTGAN and PATE-GAN.

## 🚀 Features

- **Multiple Generative Models**: GANBLR (GAN-based Bayesian Learning Rules), CTGAN (conditional tabular GAN) and PATE-GAN (differentially private GAN), plus experimental transformer- and diffusion-based generators
- **Automated Pipeline**: End-to-end training, generation, and evaluation workflows
- **TSTR Evaluation**: Train on Synthetic, Test on Real data evaluation methodology
- **Data Preprocessing**: Automated tabular preprocessing (discretization and encoding)
- **Cross-Validation Support**: Robust model validation capabilities
- **Extensible Architecture**: Easy to add new models and evaluation metrics

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Prerequisites

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11.x (strictly required due to TensorFlow compatibility) — via [pyenv](https://github.com/pyenv/pyenv) (macOS/Linux) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (Windows), or any other installer
- **[Poetry](https://python-poetry.org/docs/#installation)**: `curl -sSL https://install.python-poetry.org | python3 -`
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional, for GReaT model training)

## 📦 Installation

```bash
git clone https://github.com/datascience-works/Katabatic.git
cd katabatic
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
|----------|---------|
| Core only | `pip install katabatic` or `poetry install` |
| GANBLR (supported) | `pip install katabatic[ganblr]` or `poetry install -E ganblr` |
| CTGAN (supported) | `pip install katabatic[ctgan]` or `poetry install -E ctgan` |
| PATE-GAN (supported) | `pip install katabatic[pategan]` or `poetry install -E pategan` |
| TSTR + XGBoost | `pip install katabatic[eval]` or `poetry install -E eval` |
| Development | `poetry install --with dev` |
| All optional deps | `pip install katabatic[all]` |

Experimental models (`great`, `tabsyn`, `tabddpm`, `codi`, `medgan`, etc.) are documented in [docs/EXPERIMENTAL_MODELS.md](docs/EXPERIMENTAL_MODELS.md).
For contributor work: `poetry install --with dev -E ganblr -E ctgan -E pategan -E eval && poetry env activate`.

For GPU-accelerated GReaT training: `poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

### Verify Installation

```bash
python -c "import katabatic; print(katabatic.__version__)"
python -c "from katabatic.models.registry import ModelRegistry; print(ModelRegistry.get_supported_models())"
```

## 🚀 Quick Start

### Artifact pipeline (recommended)

Versioned datasets, models, and evaluations under `artifacts/`. See [GANBLR_FLOW.md](GANBLR_FLOW.md) for details.

```python
from katabatic.artifacts import LocalArtifactStore
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.utils.preprocess import preprocess_tabular

preprocess_tabular("raw_data/car.csv", "preprocessed_data/car.csv")

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

### Jupyter Notebook

For interactive development, launch Jupyter:

```bash
# Start Jupyter Lab
poetry run jupyter lab

# Or Jupyter Notebook
poetry run jupyter notebook
```

See `example.ipynb` for a complete walkthrough.

## 📖 Usage

### Data Preprocessing

Katabatic requires discrete/categorical data. Use the built-in preprocessing utilities:

```python
from katabatic.utils.preprocess import preprocess_tabular

# Discretize numerical features and encode categorical ones
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
synthetic_data = model.sample(size=1000)
```

#### GReaT Model

```python
from katabatic.models.great.models import GReaT
import pandas as pd

# Load your data
data = pd.read_csv("path/to/your_data.csv")

# Initialize and train model
model = GReaT(
    llm='gpt-2',  # or 'microsoft/DialoGPT-medium'
    epochs=100,
    batch_size=8
)

trainer = model.fit(data)

# Generate synthetic data
synthetic_data = model.sample(
    n_samples=1000,
    temperature=0.7
)
```

### Pipeline Usage

Katabatic provides automated pipelines for complete workflows:

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.ganblr.models import GANBLR

# Create pipeline with GANBLR
pipeline = TrainTestSplitPipeline(model=GANBLR())

# Run complete workflow: split preprocessed CSV -> train model -> TSTR evaluation.
# Legacy mode: ``real_test_dir`` defaults to ``output_dir`` (where split_dataset
# writes ``x_test.csv`` / ``y_test.csv``). ``synthetic_dir`` defaults to
# ``synthetic/<basename(output_dir)>/<model_slug>/`` if omitted.
results = pipeline.run(
    input_csv='path/to/preprocessed_data.csv',
    output_dir='output/directory',
)
# Optional overrides:
#   synthetic_dir='...', real_test_dir='...'
# ``results`` is a dict with ``message``, ``output_dir``, ``synthetic_dir``,
# ``real_test_dir``, ``tstr_results``, and ``pipeline.last_model`` is the fitted instance.
```

## 🤖 Models

### GANBLR (GAN-based Bayesian Learning Rules)

- **Type**: GAN-based generative model
- **Best for**: Discrete/categorical tabular data
- **Features**:
  - k-dependence Bayesian Networks
  - Adversarial training
  - High-quality discrete data generation

### GReaT (Generation of Realistic Tabular Data)

- **Type**: Transformer-based generative model
- **Best for**: Mixed data types (numerical + categorical)
- **Features**:
  - Pre-trained language model fine-tuning
  - Conditional generation
  - Data imputation capabilities

## 📊 Datasets

Models are benchmarked against five datasets in the data catalogue — see
[katabatic/datasets/README.md](katabatic/datasets/README.md) for details on each.

## 📊 Evaluation

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

**Statistical fidelity** (marginal JSD/KLD, DCR) is available via `katabatic.evaluate.fidelity.evaluation.StatisticalFidelityEvaluation` in artifact pipeline runs.

## 🛠 Development

### Recommended VS Code Extensions

```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension charliermarsh.ruff
code --install-extension ms-toolsai.jupyter
```

### Development Setup

```bash
git clone https://github.com/datascience-works/Katabatic.git
cd Katabatic

poetry install --with dev -E ganblr -E eval   # add -E {model} as needed

poetry check
poetry run ruff check katabatic tests
poetry run pytest                              # fast unit tests
poetry run pytest -m integration               # after installing model extras
poetry run mypy katabatic/                     # optional
```

### Project Structure

```
Katabatic/
├── katabatic/                 # Installable package (PyPI wheel)
│   ├── models/                # GANBLR, CTGAN, PATE-GAN, experimental generators
│   ├── pipeline/              # TrainTestSplitPipeline, cross-validation
│   ├── evaluate/              # TSTR, statistical fidelity
│   ├── artifacts/             # Versioned store helpers
│   └── utils/                 # preprocess, split_dataset, ...
├── artifacts/                 # Local run outputs (gitignored)
├── docs/                      # EXPERIMENTAL_MODELS.md, etc.
├── examples/                  # Notebooks per model
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

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for the development guide
(architecture, adding pipelines/evaluations, testing) and [MODEL_CONTRIBUTIONS.md](MODEL_CONTRIBUTIONS.md)
for adding or promoting a model. In short:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`, following
   [Conventional Commits](https://www.conventionalcommits.org/) — enforced by pre-commit)
4. **Push** to the branch and **open** a Pull Request into `development`

Formatting, linting, and security scans run via `pre-commit` (`ruff format`, `ruff check`, plus
the hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml)):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GANBLR**: Based on the GAN-based Bayesian Learning Rules methodology
- **GReaT**: Implements Generation of Realistic Tabular data using transformer models
- **Contributors**: Thanks to all contributors who have helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/datascience-works/Katabatic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/datascience-works/Katabatic/discussions)
- **Email**: vikumdabare@gmail.com

## 🔗 Related Projects

- [GANBLR: A Tabular Data Generation Model (ICDM 2021)](https://ieeexplore.ieee.org/document/9679177)
- [GReaT Repository](https://github.com/kathrinse/be_great)
- [Synthetic Data Resources](https://github.com/synthetic-data-resources)

---

**Happy generating!** 🎯
