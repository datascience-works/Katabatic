# Katabatic

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue)](https://python-poetry.org/)

A comprehensive framework for synthetic tabular data generation using state-of-the-art machine learning models including GANBLR and GReaT (Generation of Realistic Tabular data).

## 🚀 Features

- **Multiple Generative Models**: Support for GANBLR (GAN-based Bayesian Learning Rules) and GReaT (transformer-based generation)
- **Automated Pipeline**: End-to-end training, generation, and evaluation workflows
- **TSTR Evaluation**: Train on Synthetic, Test on Real data evaluation methodology
- **Data Preprocessing**: Automated discretization and encoding for tabular data
- **Cross-Validation Support**: Robust model validation capabilities
- **Extensible Architecture**: Easy to add new models and evaluation metrics

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.11.x (strictly required due to TensorFlow compatibility)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for GReaT model)

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

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/katabatic.git
cd katabatic
```

### 2. Set Python Version

```bash
# Set local Python version for this project
pyenv local 3.11.9
```

### 3. Install Dependencies

```bash
# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 4. GPU Support (Optional)

If you have an NVIDIA GPU and want to use it for GReaT model training:

```bash
# Install CUDA-compatible versions
poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

```bash
# Run a quick test
python -c "
import katabatic
from katabatic.models.ganblr.models import GANBLR
from katabatic.models.great.models import GReaT
print('Katabatic installation successful!')
"
```

## 🚀 Quick Start

### Basic Example

```python
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from utils import discretize_preprocess

# 1. Preprocess your data
dataset_path = "raw_data/car.csv"
output_path = "discretized_data/car.csv"
discretize_preprocess(dataset_path, output_path)

# 2. Create and run pipeline
input_csv = 'discretized_data/car.csv'
output_dir = 'sample_data/car'

pipeline = TrainTestSplitPipeline(model=GANBLR)
pipeline.run(
    input_csv=input_csv,
    output_dir=output_dir,
    synthetic_dir='synthetic/car/ganblr',
    real_test_dir='sample_data/car'
)
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
from utils import discretize_preprocess

# Discretize numerical features and encode categorical ones
discretize_preprocess(
    file_path="raw_data/your_dataset.csv",
    output_path="discretized_data/your_dataset.csv",
    bins=10,  # Number of bins for numerical discretization
    strategy='uniform'  # 'uniform', 'quantile', or 'kmeans'
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
pipeline = TrainTestSplitPipeline(model=GANBLR)

# Run complete workflow: split data -> train model -> evaluate
results = pipeline.run(
    input_csv='path/to/data.csv',
    output_dir='output/directory',
    synthetic_dir='synthetic/data/location',
    real_test_dir='test/data/location'
)
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

## 🛠 Development

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
│   ├── models/                   # Generative models
│   │   ├── ganblr/              # GANBLR implementation
│   │   └── great/               # GReaT implementation
│   ├── pipeline/                # Training pipelines
│   ├── evaluate/                # Evaluation methods
│   └── utils/                   # Utility functions
├── raw_data/                    # Raw datasets
├── sample_data/                 # Processed sample data
├── synthetic/                   # Generated synthetic data
├── Results/                     # Evaluation results
├── example.ipynb               # Usage examples
├── utils.py                    # Data preprocessing utilities
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

### Building from Source

```bash
# Build package
poetry build

# Install locally
pip install dist/katabatic-*.whl
```

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GANBLR**: Based on the GAN-based Bayesian Learning Rules methodology
- **GReaT**: Implements Generation of Realistic Tabular data using transformer models
- **Contributors**: Thanks to all contributors who have helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/katabatic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/katabatic/discussions)
- **Email**: vikumdabare@gmail.com

## 🔗 Related Projects

- [GANBLR Original Paper](https://link-to-paper)
- [GReaT Repository](https://github.com/kathrinse/be_great)
- [Synthetic Data Resources](https://github.com/synthetic-data-resources)

---

**Happy generating!** 🎯
