# Katabatic

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

A comprehensive framework for synthetic tabular data generation and evaluation. Includes 8 generative models (CTGAN, CoDi, TabDDPM, GANBLR, GReaT, Tabsyn, MedGAN, PATEGAN) and a 6-dimension evaluation pipeline that scores every model on Fidelity, Utility, Diversity, Privacy, Consistency and Stability.

## Features

- **8 Generative Models**: CTGAN, CoDi, TabDDPM, GANBLR, GReaT, Tabsyn, MedGAN, PATEGAN
- **6-Dimension Evaluation**: Fidelity, Utility, Diversity, Privacy, Consistency, Stability — combined into a single weighted composite score
- **Automated Benchmark Runner**: End-to-end pipeline — preprocess → split → train → generate → evaluate → save report
- **Data Preprocessing**: Automated encoding for mixed-type tabular data (numerical + categorical)
- **Model Registry**: Dynamic model loading with optional per-model extra dependencies
- **Extensible Architecture**: Easy to add new models via `python scaffold.py init-model <name>`

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Architecture

For a full overview of the project structure, data flow, and component relationships see [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11.x (strictly required due to TensorFlow compatibility)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Required Tools

#### 1. Python Version Management with pyenv

**macOS (via Homebrew):**

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv
brew install pyenv

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Restart shell or source profile
source ~/.zshrc
```

**Linux (Ubuntu/Debian):**

```bash
# Install dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Install pyenv
curl https://pyenv.run | bash

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Restart shell
exec "$SHELL"
```

#### 2. Install Python 3.11

```bash
# Install Python 3.11 using pyenv
pyenv install 3.11.9
pyenv global 3.11.9

# Verify installation
python --version  # Should output: Python 3.11.9
```

#### 3. Package Management with Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to your shell profile)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/katabatic.git
cd katabatic
```

### 2. Set Python Version

This project requires **Python 3.11.9 strictly**. Other versions will not work due to TensorFlow and dependency constraints.

**Check your Python version:**
```bash
python --version
```

**If you have pyenv (Mac/Linux):**
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python --version  # Should output: Python 3.11.9
```

**If you have Python 3.11.9 installed directly (Windows):**
```bash
# Tell Poetry to use it explicitly
poetry env use 3.11.9

# Verify Poetry picked it up
poetry env info
```

### 3. Install Dependencies

```bash
# Core dependencies (always required — includes the evaluation pipeline)
poetry install

# Add extras for the models you want to use
poetry install -E ctgan       # CTGAN, CoDi, MedGAN
poetry install -E tabddpm     # TabDDPM
poetry install -E ganblr      # GANBLR
poetry install -E great       # GReaT
poetry install -E all         # install everything
```

### 4. Activate the Virtual Environment

```bash
# Get the activation path
poetry env activate

# Copy the path it returns and run it, for example:
# Mac/Linux:
source /path/to/virtualenvs/katabatic-xxx-py3.11/bin/activate

# Windows:
C:\path\to\virtualenvs\katabatic-xxx-py3.11\Scripts\activate
```

### 5. GPU Support (Optional)

If you have an NVIDIA GPU and want to use it for GReaT or other torch-based models:

```bash
poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. Verify Installation

```bash
python -c "
import pandas, numpy, scipy, sklearn
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline
print('Katabatic installation successful!')
"
```

## Quick Start

### Run an existing example script

Ready-to-run scripts are available in `benchmarks/examples/` for two datasets:

| Script | Model | Dataset |
|---|---|---|
| `benchmarks/examples/run_ctgan_adult.py` | CTGAN | Adult Income |
| `benchmarks/examples/run_codi_adult.py` | CoDi | Adult Income |
| `benchmarks/examples/run_tabddpm_adult.py` | TabDDPM | Adult Income |
| `benchmarks/examples/run_ctgan_bank_marketing.py` | CTGAN | Bank Marketing |

Each script runs the full pipeline: preprocess → split → train → generate → evaluate → save report.

```bash
# place your dataset CSV in datasets/ then run
python benchmarks/examples/run_ctgan_adult.py
```

### Add a new dataset and model

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # add benchmarks/ to path

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.ctgan.models import CTGANModel

config = RunConfig(
    dataset_name     = "your_dataset",        # matches datasets/your_dataset.csv
    model_name       = "ctgan",
    categorical_cols = ['workclass', 'education'],   # actual column names
    continuous_cols  = ['age', 'fnlwgt'],
    target_col_raw   = "income",              # original target column name
    constraints      = {'age': (17, 90)},     # optional logical bounds per column
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

model = CTGANModel(epochs=100, batch_size=512, seed=42)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols,
                                continuous_cols=config.continuous_cols)

synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
```

## Usage

### Data Preprocessing

Katabatic provides a preprocessing utility that cleans a raw CSV while preserving original column names and data types. It fills numerical NaN values with the column median, drops constant/all-NaN columns, and moves the target column to the last position. Categorical columns are **not** encoded — each model receives the cleaned data and handles its own encoding internally:

```python
from katabatic.utils.preprocess import encode_preprocess

encode_preprocess(
    file_path="datasets/your_dataset.csv",
    output_path="benchmarks/processed/your_dataset_processed.csv",
    target_col="income",   # optional — moved to last column if provided
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

Evaluate any synthetic dataset with the 6-dimension evaluation pipeline:

```python
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline

pipeline = SyntheticEvaluationPipeline(
    categorical_cols=['workclass', 'education', 'occupation'],
    continuous_cols=['age', 'fnlwgt', 'capital-gain'],
)

report = pipeline.run(
    real_data=train_df,
    synthetic_data=synthetic_df,
    target_col='income',
    test_data=test_df,                  # held-out set — used by Utility for fair TSTR/TRTR comparison
    constraints={'age': (17, None)},    # optional logical bounds — (min, max)
    output_dir='benchmarks/results/my_run/',
)

print(report.composite_score)          # weighted composite 0–1
print(report.dimension_scores)         # per-dimension breakdown
```

## Models

| Model | Extra | Type | Best for |
|---|---|---|---|
| **CTGAN** | `ctgan` | GAN | Mixed tabular data |
| **CoDi** | `codi` | Diffusion + GAN | Mixed tabular data |
| **TabDDPM** | `tabddpm` | Diffusion | Numerical + categorical |
| **GANBLR** | `ganblr` | GAN + Bayesian network | Discrete/categorical data |
| **GReaT** | `great` | Transformer (LLM) | Mixed data types |
| **Tabsyn** | `tabsyn` | VAE + Diffusion | High-fidelity mixed data |
| **MedGAN** | `medgan` | GAN | Medical / binary data |
| **PATEGAN** | `pategan` | GAN + PATE | Privacy-preserving synthesis |

Install extras as needed:

```bash
poetry install -E ctgan       # CTGAN, CoDi, MedGAN
poetry install -E tabddpm     # TabDDPM
poetry install -E ganblr      # GANBLR
poetry install -E great       # GReaT
poetry install -E all         # all models
```

### Adding a new model

Use the scaffold tool to generate a boilerplate model stub:

```bash
python scaffold.py init-model mymodel dep1 dep2
```

This creates `katabatic/models/mymodel/` with the standard `Model` interface pre-filled.

## Evaluation

Katabatic evaluates synthetic data across 6 dimensions, each producing a score between 0 and 1. They are combined into a single **composite score** using fixed weights:

| Dimension | Weight | What it measures |
|---|---|---|
| **Utility** | 35% | TSTR vs TRTR accuracy gap — how useful the synthetic data is for ML |
| **Fidelity** | 25% | Statistical similarity to real data (distributions, correlations) |
| **Privacy** | 15% | Nearest-neighbour distance ratio — protection against re-identification |
| **Diversity** | 10% | Coverage of the feature space (bin coverage + Gower distance spread) |
| **Consistency** | 10% | Label coherence + constraint satisfaction across folds |
| **Stability** | 5% | Reproducibility of the model across different random seeds |

```python
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline

pipeline = SyntheticEvaluationPipeline(
    dimensions=['fidelity', 'utility', 'diversity', 'privacy', 'consistency'],
    categorical_cols=['workclass', 'education'],
    continuous_cols=['age', 'fnlwgt', 'capital-gain'],
)
report = pipeline.run(real_data=train_df, synthetic_data=synth_df, target_col='income')
print(f"Composite score: {report.composite_score:.4f}")
```

Reports are saved as JSON + CSV to the `output_dir` you specify.

## Development

### Recommended VS Code Extensions

```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.flake8
code --install-extension ms-python.black-formatter
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.isort
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/katabatic.git
cd katabatic

# Install development dependencies
poetry install --group dev

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy katabatic/
```

### Project Structure

```
katabatic/
├── katabatic/                    # Main package
│   ├── models/                   # Generative models (8 implementations)
│   │   ├── base_model.py        # Abstract Model interface
│   │   ├── registry.py          # Dynamic model loader
│   │   ├── ctgan/
│   │   ├── codi/
│   │   ├── tabddpm/
│   │   ├── ganblr/
│   │   ├── great/
│   │   ├── tabsyn/
│   │   ├── medgan/
│   │   └── pategan/
│   ├── pipeline/
│   │   └── evaluation_pipeline.py   # SyntheticEvaluationPipeline
│   ├── evaluate/                # One subpackage per dimension
│   │   ├── fidelity/
│   │   ├── utility/
│   │   ├── diversity/
│   │   ├── privacy/
│   │   ├── consistency/
│   │   ├── stability/
│   │   └── report/              # EvaluationReport + composite score
│   └── utils/
│       ├── preprocess.py        # encode_preprocess + mappings
│       ├── split_dataset.py     # stratified train/test split
│       └── column_types.py      # categorical/continuous auto-detection
├── benchmarks/
│   ├── runner.py                # RunConfig + pipeline orchestration helpers
│   ├── examples/                # Ready-to-run benchmark scripts
│   │   ├── run_ctgan_adult.py
│   │   ├── run_codi_adult.py
│   │   ├── run_tabddpm_adult.py
│   │   └── run_ctgan_bank_marketing.py
│   ├── processed/               # Preprocessed CSVs (git-ignored)
│   ├── splits/                  # Train/test splits (git-ignored)
│   ├── synthetic/               # Generated synthetic data (git-ignored)
│   └── results/                 # Evaluation reports (git-ignored)
├── datasets/                    # Raw dataset CSVs (git-ignored)
├── scaffold.py                  # CLI — scaffold new model stubs
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

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards & Style Guide

We maintain high code quality standards to ensure consistency, readability, and maintainability across the codebase.

#### Python Style Guidelines

- **PEP 8 Compliance**: All code must follow [PEP 8](https://pep8.org/) style guidelines
- **Line Length**: Maximum 88 characters (Black's default)
- **Imports**: Use `isort` for import organization
- **Type Hints**: Add type hints for all public functions and class methods
- **Docstrings**: Include docstrings for all modules, classes, and functions using Google or NumPy style

#### Code Formatting with autopep8

We use `autopep8` as our primary code formatter to ensure consistent code style:

```bash
# Install autopep8 (included in dev dependencies)
poetry add --group dev autopep8

# Format a single file
poetry run autopep8 --in-place --aggressive --aggressive your_file.py

# Format entire project
poetry run autopep8 --in-place --aggressive --aggressive --recursive .

# Check formatting without making changes
poetry run autopep8 --diff --aggressive --aggressive --recursive .
```

#### Recommended autopep8 Configuration

Create a `.autopep8` configuration file in the project root:

```ini
# .autopep8
[autopep8]
max_line_length = 88
ignore = E203,W503
aggressive = 2
recursive = true
```

#### Additional Formatting Tools

While autopep8 is our primary formatter, you may also use these complementary tools:

```bash
# isort for import sorting
poetry run isort .

# Black as an alternative formatter (if preferred)
poetry run black .

# flake8 for linting
poetry run flake8 katabatic/

# mypy for static type checking
poetry run mypy katabatic/
```

#### Pre-commit Hooks

Set up pre-commit hooks to automatically format code before commits:

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/pre-commit/mirrors-autopep8
    rev: v2.0.2
    hooks:
      - id: autopep8
        args: [--aggressive, --aggressive, --in-place]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile, black]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]
EOF

# Install the hooks
poetry run pre-commit install
```

#### VS Code Configuration

Add these settings to your VS Code workspace settings (`.vscode/settings.json`):

```json
{
  "python.formatting.provider": "autopep8",
  "python.formatting.autopep8Args": [
    "--aggressive",
    "--aggressive",
    "--max-line-length=88"
  ],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--ignore=E203,W503"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.sortImports.args": ["--profile", "black"]
}
```

#### Code Quality Checklist

Before submitting code, ensure:

- [ ] Code is formatted with autopep8: `poetry run autopep8 --diff --aggressive --aggressive --recursive .`
- [ ] Imports are sorted: `poetry run isort --check-only .`
- [ ] No linting errors: `poetry run flake8 katabatic/`
- [ ] Type hints pass checking: `poetry run mypy katabatic/`
- [ ] All tests pass: `poetry run pytest`
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventional commit format

#### Naming Conventions

- **Variables and Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Methods**: `_leading_underscore`
- **Modules**: `lowercase` or `snake_case`

#### Documentation Standards

- Use Google-style docstrings for consistency
- Include type information in docstrings when not obvious from type hints
- Provide examples for complex functions
- Update README and documentation when adding new features

**Example Docstring:**

```python
def generate_synthetic_data(
    model: BaseModel,
    n_samples: int,
    temperature: float = 0.7
) -> pd.DataFrame:
    """Generate synthetic tabular data using the specified model.

    Args:
        model: Trained generative model instance
        n_samples: Number of synthetic samples to generate
        temperature: Sampling temperature for generation (default: 0.7)

    Returns:
        DataFrame containing synthetic data samples

    Raises:
        ValueError: If model is not trained or n_samples <= 0

    Example:
        >>> model = GANBLR()
        >>> model.fit(X_train, y_train)
        >>> synthetic_data = generate_synthetic_data(model, 1000)
    """
```

#### Testing Standards

- Write unit tests for new features
- Maintain minimum 80% code coverage
- Use descriptive test names
- Include edge case testing
- Mock external dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **GANBLR**: Based on the GAN-based Bayesian Learning Rules methodology
- **GReaT**: Implements Generation of Realistic Tabular data using transformer models
- **Contributors**: Thanks to all contributors who have helped improve this project

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/katabatic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/katabatic/discussions)
- **Email**: vikumdabare@gmail.com

## Related Projects

- [GANBLR Original Paper](https://link-to-paper)
- [GReaT Repository](https://github.com/kathrinse/be_great)
- [Synthetic Data Resources](https://github.com/synthetic-data-resources)

---

**Happy generating!**
