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
│ • CTGAN         │    │ • Evaluation    │    │ • Fidelity      │
│ • GReaT         │ ───►   Pipeline     │ ───► • Utility       │
│ • CustomModel   │    │ • Custom        │    │ • Privacy       │
│ • ...8 total    │    │   Pipeline      │    │ • + 3 more      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Design Principles

1. **Extensibility**: Each component inherits from a base class with defined interfaces
2. **Modularity**: Models, pipelines, and evaluations are loosely coupled
3. **Configurability**: Pipeline configurations support different model-evaluation combinations
4. **Reproducibility**: Built-in support for seeds and experiment tracking

## 📁 Codebase Structure

```
katabatic/
├── katabatic/                    # Main package
│   ├── __init__.py
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base class for all models
│   │   ├── registry.py           # Dynamic model loader
│   │   ├── ctgan/                # CTGAN implementation
│   │   ├── codi/                 # CoDi implementation
│   │   ├── tabddpm/              # TabDDPM implementation
│   │   ├── ganblr/               # GANBLR implementation
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── kdb.py
│   │   │   ├── utils.py
│   │   │   ├── poetry.lock
│   │   │   └── pyproject.toml   # Model-specific dependencies
│   │   ├── great/                # GReaT implementation
│   │   │   ├── models.py
│   │   │   ├── great_dataset.py
│   │   │   ├── great_trainer.py
│   │   │   ├── great_utils.py
│   │   │   ├── great_start.py
│   │   │   ├── poetry.lock
│   │   │   └── pyproject.toml   # Model-specific dependencies
│   │   ├── tabsyn/               # Tabsyn implementation
│   │   ├── medgan/               # MedGAN implementation
│   │   └── pategan/              # PATEGAN implementation
│   ├── pipeline/                 # Pipeline implementations
│   │   ├── __init__.py
│   │   ├── base_pipeline.py      # Abstract base class for pipelines
│   │   └── evaluation_pipeline.py  # SyntheticEvaluationPipeline (main)
│   ├── evaluate/                 # Evaluation implementations
│   │   ├── __init__.py
│   │   ├── base_evaluation.py    # Abstract base class for evaluations
│   │   ├── fidelity/             # JSD + Wasserstein + Correlation
│   │   ├── utility/              # TSTR vs TRTR across 5 classifiers
│   │   ├── diversity/            # Category + Bin + Gower coverage
│   │   ├── privacy/              # NNDR + duplicate detection
│   │   ├── consistency/          # Discriminator + constraints + feature importance
│   │   ├── stability/            # Multi-run variance
│   │   └── report/               # EvaluationReport + composite scoring
│   └── utils/                    # Shared utilities
│       ├── column_types.py       # Categorical/continuous auto-detection
│       ├── split_dataset.py      # Stratified train/test split
│       └── preprocess.py         # encode_preprocess + data cleaning
├── benchmarks/
│   ├── runner.py                 # RunConfig + shared pipeline helpers
│   ├── splits/                   # Preprocessed train/test splits per dataset
│   ├── synthetic/                # Generated synthetic data per dataset/model
│   ├── results/                  # Evaluation results per dataset/model
│   └── examples/                 # Reference run scripts
│       ├── run_ctgan_adult.py
│       ├── run_codi_adult.py
│       ├── run_tabddpm_adult.py
│       └── run_ctgan_bank_marketing.py
├── datasets/
│   ├── adult.csv
│   └── bank_marketing.csv
├── scaffold.py                   # Scaffolds a new model (python scaffold.py init-model <name>)
├── pyproject.toml                # Main project dependencies
├── poetry.lock                   # Locked dependencies
├── Makefile                      # Build and development commands
└── README.md                     # User documentation
```

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
Raw Data → encode_preprocess → Train/Test Split → Model Training → Synthetic Generation → SyntheticEvaluationPipeline → EvaluationReport
```

## 🔄 Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd katabatic
pyenv local 3.11.9
poetry install
poetry shell

# Verify installation
python -c "from katabatic.models.ganblr.models import GANBLR; print('Setup successful')"
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

   # Run a benchmark example
   python benchmarks/examples/run_ctgan_adult.py
   ```

4. **Update Documentation**

   - Update this CONTRIBUTING.md if architecture changes
   - Update README.md for user-facing changes
   - Add docstrings and type hints

5. **Submit Pull Request**

## 🤖 Adding New Models

The recommended way to add a new model is via the scaffold tool, which generates all the boilerplate automatically:

```bash
python scaffold.py init-model your_model_name dep1 dep2
```

This creates `katabatic/models/your_model_name/` with `__init__.py`, `models.py`, and `utils.py` pre-filled, and registers the model in `katabatic/models/registry.py`.

### Step 1: Create Model Directory Structure

```bash
mkdir -p katabatic/models/your_model_name
cd katabatic/models/your_model_name
```

### Step 2: Create Required Files

```bash
touch __init__.py
touch models.py
touch utils.py  # If model-specific utilities needed
touch README.md  # Model-specific documentation
```

### Step 3: Implement Base Model Interface

Create `katabatic/models/your_model_name/models.py`:

```python
from katabatic.models.base_model import Model
import pandas as pd
import numpy as np
import os
from typing import Optional


class YourModelName(Model):
    """
    Your model description here.

    Parameters
    ----------
    param1 : type
        Description of parameter 1
    param2 : type, optional
        Description of parameter 2
    """

    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False

    def train(
        self,
        data_dir: str,
        *args,
        categorical_cols: Optional[list] = None,
        continuous_cols: Optional[list] = None,
        **kwargs,
    ) -> "YourModelName":
        """
        Train the model on the given dataset.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing x_train.csv and y_train.csv
        categorical_cols : list, optional
            Names of categorical columns
        continuous_cols : list, optional
            Names of continuous columns

        Returns
        -------
        self
        """
        x_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

        # Implement your training logic here
        self._fit_internal(x_train, y_train, **kwargs)
        self.is_fitted = True
        return self

    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        pd.DataFrame
            Generated synthetic data with original column names
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before sampling")

        synthetic_data = self._generate_samples(n_samples, **kwargs)
        return synthetic_data

    def evaluate(self, **kwargs) -> dict:
        """
        Quick TSTR evaluation. For comprehensive evaluation use
        SyntheticEvaluationPipeline instead.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        # Implement quick evaluation logic
        raise NotImplementedError

    def check_dependencies(self) -> bool:
        """Return True if all required dependencies are installed."""
        return True

    def _fit_internal(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        """Internal fitting logic - implement your algorithm here"""
        raise NotImplementedError

    def _generate_samples(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """Internal sampling logic - implement your algorithm here"""
        raise NotImplementedError
```

### Step 4: Update Model **init**.py

`katabatic/models/your_model_name/__init__.py`:

```python
from .models import YourModelName

__all__ = ['YourModelName']
```

### Step 5: Update Main Models **init**.py

Add to `katabatic/models/__init__.py`:

```python
from .your_model_name import YourModelName
```

### Step 6: Add Model Dependencies

Add your model's dependencies to the root `pyproject.toml`:

```toml
[tool.poetry.dependencies]
# Add as optional dependency
your-dep = {version = "^1.0", optional = true}

[tool.poetry.extras]
# Create an extras group for your model
your_model_name = ["your-dep"]
```

Install with:

```bash
poetry install -E your_model_name
```

### Step 7: Test Your Model

Create a benchmark script based on the existing examples:

```python
# benchmarks/examples/run_your_model_adult.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from katabatic.models.your_model_name.models import YourModelName
from benchmarks.runner import RunConfig, preprocess_and_split, save_synthetic, evaluate

config = RunConfig(
    dataset="adult",
    model_name="your_model_name",
    target_col_raw="income",
    categorical_cols=["workclass", "education", "marital-status", "occupation",
                      "relationship", "race", "gender", "native-country"],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    constraints={"age": (17, 90), "hours-per-week": (1, 99)},
)

train_df, test_df, paths = preprocess_and_split(config)

model = YourModelName()
model.train(
    data_dir=paths["splits_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

synthetic_df = model.sample(n_samples=len(train_df))
save_synthetic(synthetic_df, train_df, paths)
evaluate(train_df, test_df, synthetic_df, config, paths)
```

## 🔄 Adding New Pipelines

The main evaluation pipeline is `SyntheticEvaluationPipeline` in `katabatic/pipeline/evaluation_pipeline.py`. For most use cases you should use it directly rather than writing a new pipeline.

If you need a custom pipeline, subclass `Pipeline` from `base_pipeline.py`:

### Step 1: Create Pipeline File

```bash
touch katabatic/pipeline/your_pipeline_name.py
```

### Step 2: Implement Pipeline Class

`katabatic/pipeline/your_pipeline_name.py`:

```python
from katabatic.pipeline.base_pipeline import Pipeline
from katabatic.models.base_model import Model
from typing import Optional


class YourPipelineName(Pipeline):
    """
    Description of your pipeline.

    Parameters
    ----------
    model : Model
        The model class to use for training
    """

    def __init__(self, model: Model, **kwargs):
        super().__init__(model)

    def run(self, *args, **kwargs):
        """
        Run your pipeline with the given arguments.

        Returns
        -------
        Result of the pipeline run
        """
        # Step 1: Initialize model
        current_model = self.model()

        # Step 2: Train model
        current_model.train(kwargs["data_dir"], *args, **kwargs)

        # Step 3: Generate synthetic data
        synthetic_df = current_model.sample(kwargs.get("n_samples", 1000))

        return synthetic_df

    def __repr__(self):
        return f"YourPipelineName(model={self.model})"
```

### Step 3: Test Your Pipeline

```python
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.your_pipeline_name import YourPipelineName

pipeline = YourPipelineName(model=GANBLR)
result = pipeline.run(
    data_dir="benchmarks/splits/adult",
    n_samples=1000,
)
print(result)
```

## 📊 Adding New Evaluations

### Step 1: Create Evaluation Directory

```bash
mkdir -p katabatic/evaluate/your_evaluation_name
touch katabatic/evaluate/your_evaluation_name/__init__.py
touch katabatic/evaluate/your_evaluation_name/evaluation.py
```

### Step 2: Implement Evaluation Class

`katabatic/evaluate/your_evaluation_name/evaluation.py`:

```python
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from katabatic.evaluate.base_evaluation import Evaluation
from typing import Dict, Any, Optional


class YourEvaluationName(Evaluation):
    """
    Description of your evaluation methodology.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real training data
    synthetic_data : pd.DataFrame
        Synthetic data to evaluate
    **kwargs
        Additional evaluation parameters
    """

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs):
        super().__init__(real_data=real_data, synthetic_data=synthetic_data, **kwargs)

    def evaluate(self) -> Dict[str, Any]:
        """
        Perform the evaluation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation results
        """
        results = {}

        # Implement your evaluation logic
        models = self._get_evaluation_models()

        for model_name, model in models.items():
            metrics = self._evaluate_single_model(model, model_name)
            results[model_name] = metrics

        self._save_results(results)
        self._print_results(results)

        return results

    def _get_evaluation_models(self):
        """Define models to use for evaluation"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier

        return {
            "LogisticRegression": LogisticRegression(random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "MLP": MLPClassifier(random_state=42, max_iter=500)
        }

    def _evaluate_single_model(self, model, model_name: str) -> Dict[str, float]:
        """Evaluate a single model"""
        model.fit(self.synthetic_data.iloc[:, :-1], self.synthetic_data.iloc[:, -1])
        y_pred = model.predict(self.real_data.iloc[:, :-1])
        y_test = self.real_data.iloc[:, -1]

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save results to CSV file"""
        results_dir = os.path.join("benchmarks", "results")
        os.makedirs(results_dir, exist_ok=True)

        rows = []
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                rows.append([model_name, metric_name, round(value, 4)])

        df_results = pd.DataFrame(rows, columns=["Model", "Metric", "Value"])
        output_path = os.path.join(results_dir, "your_evaluation.csv")
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def _print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print(f"\n{self.__class__.__name__} Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
```

### Step 3: Update Evaluation **init**.py

`katabatic/evaluate/your_evaluation_name/__init__.py`:

```python
from .evaluation import YourEvaluationName

__all__ = ['YourEvaluationName']
```

### Step 4: Test Your Evaluation

```python
import pandas as pd
from katabatic.evaluate.your_evaluation_name.evaluation import YourEvaluationName

real_df = pd.read_csv("benchmarks/splits/adult/train_full.csv")
synthetic_df = pd.read_csv("benchmarks/synthetic/adult/ctgan/synthetic.csv")

evaluation = YourEvaluationName(real_data=real_df, synthetic_data=synthetic_df)
results = evaluation.evaluate()
print(results)
```

## 🧪 Testing and Quality Assurance

### Test Infrastructure

Katabatic includes a comprehensive test suite with:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **Test Fixtures**: Reusable test data and mock objects
- **Coverage Reporting**: Track test coverage metrics
- **Quality Checks**: Automated code formatting and linting

### Running Tests

**Quick Start with Make:**

```bash
# Run all tests
make test

# Run specific test types
make test-unit              # Unit tests only
make test-integration       # Integration tests only
make test-fast              # Exclude slow tests
make test-coverage          # Tests with coverage report
make test-quality           # Code quality checks

# Component-specific tests
make test-models            # Model-related tests
make test-pipelines         # Pipeline-related tests
make test-evaluations       # Evaluation-related tests
```

**Direct Pytest Usage:**

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_base_model.py

# Run tests with specific markers
pytest -m "unit"                # Unit tests only
pytest -m "integration"         # Integration tests only
pytest -m "models"              # Model tests only
pytest -m "not slow"            # Exclude slow tests

# Run with coverage
pytest --cov=katabatic --cov-report=html tests/

# Run specific test method
pytest tests/unit/test_base_model.py::TestBaseModel::test_model_is_abstract
```

### Writing Tests

**Test Structure:**

```
tests/
├── conftest.py                 # Shared fixtures and utilities
├── unit/                       # Unit tests
│   ├── test_base_model.py     # Base model tests
│   ├── test_base_pipeline.py  # Base pipeline tests
│   ├── test_ganblr_model.py   # GANBLR model tests
│   └── test_utils.py          # Utility function tests
├── integration/                # Integration tests
│   ├── test_pipeline_integration.py
│   └── test_end_to_end.py
└── fixtures/                   # Test data files
```

**Writing Unit Tests:**

```python
import pytest
from katabatic.models.your_model_name.models import YourModelName

@pytest.mark.unit
@pytest.mark.models
class TestYourModel:
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = YourModelName()
        assert isinstance(model, YourModelName)
        assert model.is_fitted is False

    def test_model_training(self, sample_dataset_files):
        """Test model training with mock data."""
        model = YourModelName()

        model.train(
            data_dir=sample_dataset_files['dir'],
            categorical_cols=sample_dataset_files['categorical_cols'],
            continuous_cols=sample_dataset_files['continuous_cols'],
        )

        assert model.is_fitted is True
```

**Writing Integration Tests:**

```python
@pytest.mark.integration
@pytest.mark.pipeline
class TestYourModelIntegration:
    def test_model_with_pipeline(self, tmp_path, sample_splits_dir):
        """Test model trains, samples, and evaluates end-to-end."""
        from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline
        import pandas as pd

        model = YourModelName()
        model.train(data_dir=sample_splits_dir, categorical_cols=[...], continuous_cols=[...])
        synthetic_df = model.sample(n_samples=100)

        train_df = pd.read_csv(f"{sample_splits_dir}/train_full.csv")
        pipeline = SyntheticEvaluationPipeline(categorical_cols=[...], continuous_cols=[...])
        report = pipeline.run(real_data=train_df, synthetic_data=synthetic_df, target_col="target")

        assert report.composite_score >= 0.0
        assert os.path.exists(os.path.join(tmp_path, "synthetic.csv"))
```

**Available Test Fixtures:**

```python
def test_with_fixtures(sample_binary_dataset, sample_dataset_files, temp_dir, mock_model):
    # sample_binary_dataset: (X, y) tuple with mixed feature types
    # sample_dataset_files: Dict with paths to CSV files
    # temp_dir: Temporary directory path (auto-cleanup)
    # mock_model: MockModel instance for testing pipelines
```

### Test Requirements for New Components

**For New Models:**

1. Test inheritance from base Model class
2. Test initialization and parameter handling
3. Test train/sample/evaluate method interfaces
4. Test integration with `SyntheticEvaluationPipeline`
5. Test error handling and edge cases

**For New Pipelines:**

1. Test initialization with models
2. Test run method with various parameters
3. Test file I/O operations
4. Test integration with evaluation
5. Test error conditions and recovery

**For New Evaluations:**

1. Test evaluation metric calculations
2. Test data loading and preprocessing
3. Test result formatting and saving
4. Test integration with `SyntheticEvaluationPipeline`
5. Test handling of edge cases

### Code Quality Checks

```bash
# Run all quality checks
make test-quality

# Or individually
black --check katabatic/ tests/      # Code formatting
isort --check-only katabatic/ tests/ # Import sorting
flake8 katabatic/                    # Linting
mypy katabatic/                      # Type checking
```

**Quality Standards:**

- **Code Coverage**: Minimum 80% for new code
- **Code Formatting**: Use Black with 88-character line limit
- **Import Sorting**: Use isort with Black profile
- **Linting**: Pass flake8 checks
- **Type Hints**: Add type hints for public interfaces

```bash
# Format code
black katabatic/
isort katabatic/

# Lint code
flake8 katabatic/
pylint katabatic/

# Type checking
mypy katabatic/
```

### Integration Testing

Create integration tests for new components:

```python
# tests/integration/test_new_model.py
import pytest
import os
import pandas as pd
from katabatic.models.your_model_name.models import YourModelName
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline


def test_new_model_integration(tmp_path):
    """Test new model with evaluation pipeline end-to-end"""
    splits_dir = "benchmarks/splits/adult"
    train_df = pd.read_csv(f"{splits_dir}/train_full.csv")

    model = YourModelName()
    model.train(
        data_dir=splits_dir,
        categorical_cols=["workclass", "education"],
        continuous_cols=["age", "fnlwgt"],
    )
    synthetic_df = model.sample(n_samples=100)

    pipeline = SyntheticEvaluationPipeline(
        categorical_cols=["workclass", "education"],
        continuous_cols=["age", "fnlwgt"],
    )
    report = pipeline.run(real_data=train_df, synthetic_data=synthetic_df, target_col="income")

    assert report.composite_score >= 0.0
```

## 📖 Usage Examples

### Example 1: Basic Model Usage via Runner

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from katabatic.models.ctgan.models import CTGANModel
from benchmarks.runner import RunConfig, preprocess_and_split, save_synthetic, evaluate

config = RunConfig(
    dataset="adult",
    model_name="ctgan",
    target_col_raw="income",
    categorical_cols=["workclass", "education", "marital-status", "occupation",
                      "relationship", "race", "gender", "native-country"],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    constraints={"age": (17, 90), "hours-per-week": (1, 99)},
)

train_df, test_df, paths = preprocess_and_split(config)

model = CTGANModel(epochs=300)
model.train(
    data_dir=paths["splits_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

synthetic_df = model.sample(n_samples=len(train_df))
save_synthetic(synthetic_df, train_df, paths)
evaluate(train_df, test_df, synthetic_df, config, paths)
```

### Example 2: Direct Pipeline Usage

```python
import pandas as pd
from katabatic.models.ctgan.models import CTGANModel
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline

categorical_cols = ["workclass", "education", "marital-status"]
continuous_cols = ["age", "fnlwgt", "hours-per-week"]

model = CTGANModel(epochs=300)
model.train(
    data_dir="benchmarks/splits/adult",
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
)
synthetic_df = model.sample(n_samples=1000)

train_df = pd.read_csv("benchmarks/splits/adult/train_full.csv")

pipeline = SyntheticEvaluationPipeline(
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
)
report = pipeline.run(real_data=train_df, synthetic_data=synthetic_df, target_col="income")
print(f"Composite score: {report.composite_score:.4f}")
print(report.dimension_scores)
```

### Example 3: Standalone Model Usage

```python
import pandas as pd
from katabatic.models.ganblr.models import GANBLR

# Load preprocessed splits
x_train = pd.read_csv("benchmarks/splits/adult/x_train.csv")
y_train = pd.read_csv("benchmarks/splits/adult/y_train.csv").squeeze()

# Train model directly
model = GANBLR()
model.train(data_dir="benchmarks/splits/adult")

# Generate synthetic data
synthetic_df = model.sample(n_samples=1000)
print(f"Generated {len(synthetic_df)} synthetic samples")

# Evaluate model
results = model.evaluate()
print(f"TSTR results: {results}")
```

### Example 4: Multi-Model Comparison

```python
import pandas as pd
from katabatic.models.ctgan.models import CTGANModel
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline

splits_dir = "benchmarks/splits/adult"
train_df = pd.read_csv(f"{splits_dir}/train_full.csv")

categorical_cols = ["workclass", "education", "marital-status"]
continuous_cols = ["age", "fnlwgt", "hours-per-week"]

models = {
    "ctgan": CTGANModel(epochs=300),
    "ganblr": GANBLR(),
}

pipeline = SyntheticEvaluationPipeline(
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
)

for name, model in models.items():
    model.train(data_dir=splits_dir, categorical_cols=categorical_cols, continuous_cols=continuous_cols)
    synthetic_df = model.sample(n_samples=len(train_df))
    report = pipeline.run(real_data=train_df, synthetic_data=synthetic_df, target_col="income")
    print(f"{name}: composite_score={report.composite_score:.4f}")
```

### Example 5: Jupyter Notebook Workflow

```python
# Cell 1: Setup and imports
import pandas as pd
from katabatic.models.ctgan.models import CTGANModel
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline
from katabatic.utils.preprocess import encode_preprocess
from katabatic.utils.split_dataset import split_dataset

# Cell 2: Preprocess raw data (only needed once — skipped automatically on re-runs)
from benchmarks.runner import RunConfig, preprocess_and_split

config = RunConfig(
    dataset="adult",
    model_name="ctgan",
    target_col_raw="income",
    categorical_cols=["workclass", "education", "marital-status"],
    continuous_cols=["age", "fnlwgt", "hours-per-week"],
)
train_df, test_df, paths = preprocess_and_split(config)

# Cell 3: Run CTGAN
model_ctgan = CTGANModel(epochs=300)
model_ctgan.train(data_dir=paths["splits_dir"], categorical_cols=config.categorical_cols)
synthetic_ctgan = model_ctgan.sample(n_samples=len(train_df))

# Cell 4: Run GANBLR
model_ganblr = GANBLR()
model_ganblr.train(data_dir=paths["splits_dir"])
synthetic_ganblr = model_ganblr.sample(n_samples=len(train_df))

# Cell 5: Compare results
pipeline = SyntheticEvaluationPipeline(
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

for name, synthetic_df in [("ctgan", synthetic_ctgan), ("ganblr", synthetic_ganblr)]:
    report = pipeline.run(real_data=train_df, synthetic_data=synthetic_df, target_col="income")
    print(f"{name}: {report.composite_score:.4f} | {report.dimension_scores}")
```

## 🏆 Best Practices

### Code Style

1. **Follow PEP 8**: Use consistent formatting and naming conventions
2. **Type Hints**: Add type hints to all public methods
3. **Docstrings**: Use NumPy-style docstrings for all classes and methods
4. **Error Handling**: Include proper error handling and validation

### Architecture Guidelines

1. **Single Responsibility**: Each class should have one clear responsibility
2. **Dependency Injection**: Models should be injected into pipelines
3. **Configuration**: Use parameters rather than hard-coded values
4. **Logging**: Add appropriate logging for debugging and monitoring

### Performance Considerations

1. **Memory Management**: Be mindful of memory usage with large datasets
2. **Vectorization**: Use NumPy/Pandas vectorized operations when possible
3. **Caching**: Cache expensive computations when appropriate
4. **Progress Tracking**: Add progress bars for long-running operations

### Testing Guidelines

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Edge Cases**: Test boundary conditions and error cases
4. **Reproducibility**: Use fixed random seeds in tests

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Wrong way
from katabatic.models.ganblr import GANBLR

# Correct way
from katabatic.models.ganblr.models import GANBLR
```

#### 2. Missing Dependencies

```bash
# Install model-specific dependencies via extras
poetry install -E ctgan
poetry install -E ganblr
poetry install -E pategan
```

#### 3. Data Format Issues

```python
# Ensure proper data types
y = pd.read_csv("y_train.csv").squeeze()  # Convert to Series
X = pd.read_csv("x_train.csv")            # Keep as DataFrame
```

#### 4. Path Issues

```python
# Use absolute paths or os.path.join
import os
splits_dir = os.path.join("benchmarks", "splits", "adult")
```

### Debugging Tips

1. **Enable Verbose Logging**:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Data Shapes**:

   ```python
   print(f"X shape: {X.shape}, y shape: {y.shape}")
   ```

3. **Validate Paths**:
   ```python
   import os
   assert os.path.exists(splits_dir), f"Directory not found: {splits_dir}"
   ```

### Getting Help

1. **Check Error Messages**: Read the full stack trace
2. **Review Documentation**: Ensure you're following the correct API
3. **Check Examples**: Compare with working examples in `benchmarks/examples/`
4. **Create Minimal Reproduction**: Isolate the issue with minimal code

## 🚀 Advanced Development Topics

### Adding Model-Specific Optimizations

For models with special requirements:

```python
class OptimizedModel(Model):
    def train(self, data_dir: str, *, categorical_cols=None, continuous_cols=None, **kwargs):
        # Adjust parameters based on dataset size
        x_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
        n_rows = len(x_train)

        if n_rows > 10_000:
            self.batch_size = 256
            self.epochs = 50
        else:
            self.batch_size = 64
            self.epochs = 200

        # Continue with training...
```

### Custom Data Loaders

For specialized data handling:

```python
class CustomDataLoader:
    def __init__(self, splits_dir: str):
        self.splits_dir = splits_dir

    def load(self):
        x_train = pd.read_csv(os.path.join(self.splits_dir, "x_train.csv"))
        y_train = pd.read_csv(os.path.join(self.splits_dir, "y_train.csv")).squeeze()
        return x_train, y_train
```

### Experiment Tracking Integration

```python
import wandb

class TrackedModel(Model):
    def train(self, data_dir: str, *, categorical_cols=None, continuous_cols=None, **kwargs):
        wandb.init(project="katabatic-experiments")
        # Log parameters and metrics
        wandb.log({"epoch": epoch, "loss": loss})
```

---

This development guide should serve as your comprehensive reference for contributing to the Katabatic framework. For questions or clarifications, please reach out to the development team or create an issue in the repository.
