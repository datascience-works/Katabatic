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

```
katabatic/
├── katabatic/                    # Main package
│   ├── __init__.py
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base class for all models
│   │   ├── ganblr/               # GANBLR model implementation
│   │   │   ├── __init__.py
│   │   │   ├── models.py         # Main GANBLR class
│   │   │   ├── kdb.py           # K-Dependence Bayesian classifier
│   │   │   ├── utils.py         # GANBLR-specific utilities
│   │   │   ├── poetry.lock      # Model-specific dependencies
│   │   │   └── pyproject.toml   # Model configuration
│   │   └── great/               # GReaT model implementation
│   │       ├── models.py        # Main GReaT class
│   │       ├── great_dataset.py # Dataset handling
│   │       ├── great_trainer.py # Training logic
│   │       ├── great_utils.py   # GReaT-specific utilities
│   │       ├── great_start.py   # Sampling initialization
│   │       ├── poetry.lock      # Model-specific dependencies
│   │       └── pyproject.toml   # Model configuration
│   ├── pipeline/                 # Pipeline implementations
│   │   ├── __init__.py
│   │   ├── base_pipeline.py      # Abstract base class for pipelines
│   │   ├── train_test_split/     # Train-test split pipeline
│   │   │   └── pipeline.py
│   │   └── cross_validation/     # Cross-validation pipeline
│   │       └── pipeline.py
│   ├── evaluate/                 # Evaluation implementations
│   │   ├── __init__.py
│   │   ├── base_evaluation.py    # Abstract base class for evaluations
│   │   └── tstr/                # TSTR evaluation
│   │       └── evaluation.py
│   ├── utils/                    # Shared utilities
│   │   └── split_dataset.py     # Dataset splitting utilities
│   └── synthetic/               # Generated synthetic data storage
├── raw_data/                    # Original datasets
├── discretized_data/            # Preprocessed datasets
├── sample_data/                 # Train/test splits organized by dataset
│   ├── adult/
│   ├── car/
│   └── [other_datasets]/
├── synthetic/                   # Synthetic data organized by dataset/model
│   └── car/
│       └── ganblr/
├── Results/                     # Evaluation results
├── utils.py                     # Top-level utility functions
├── main.py                      # Command-line interface
├── example.ipynb               # Usage examples
├── pyproject.toml              # Main project dependencies
├── poetry.lock                 # Locked dependencies
├── Makefile                    # Build and development commands
└── README.md                   # User documentation
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
Raw Data → Preprocessing → Train/Test Split → Model Training → Synthetic Generation → Evaluation → Results
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

   # Test with example notebook
   jupyter lab example.ipynb
   ```

4. **Update Documentation**

   - Update this CONTRIBUTING.md if architecture changes
   - Update README.md for user-facing changes
   - Add docstrings and type hints

5. **Submit Pull Request**

## 🤖 Adding New Models

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
from typing import Union, Optional, Any


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

    def __init__(self, param1: Any, param2: Optional[Any] = None):
        self.param1 = param1
        self.param2 = param2
        self._is_fitted = False

    def train(self, dataset: str, size_category: str = 'small', *args, **kwargs) -> None:
        """
        Train the model on the given dataset.

        Parameters
        ----------
        dataset : str
            Path to the dataset directory containing train/test splits
        size_category : str, default='small'
            Dataset size category for optimization
        *args, **kwargs
            Additional training parameters

        Note
        ----
        Expected dataset structure:
        dataset/
        ├── x_train.csv
        ├── y_train.csv
        ├── x_test.csv
        └── y_test.csv
        """
        # Load training data
        x_train = pd.read_csv(f"{dataset}/x_train.csv")
        y_train = pd.read_csv(f"{dataset}/y_train.csv").values.ravel()

        # Implement your training logic here
        # Example:
        self._fit_internal(x_train, y_train, **kwargs)
        self._is_fitted = True

        # Save synthetic data
        synthetic_x, synthetic_y = self._generate_synthetic_data(len(x_train))
        self._save_synthetic_data(synthetic_x, synthetic_y, kwargs.get('synthetic_dir'))

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, **kwargs) -> float:
        """
        Evaluate the model using TSTR methodology.

        Parameters
        ----------
        x_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        **kwargs
            Additional evaluation parameters

        Returns
        -------
        float
            Evaluation score (e.g., accuracy)
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before evaluation")

        # Implement your evaluation logic
        # This typically involves:
        # 1. Generate synthetic data
        # 2. Train a classifier on synthetic data
        # 3. Test on real data
        # 4. Return performance metric

        return score

    def sample(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate synthetic samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        **kwargs
            Additional sampling parameters

        Returns
        -------
        np.ndarray
            Generated synthetic data
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before sampling")

        # Implement your sampling logic
        synthetic_data = self._generate_samples(n_samples, **kwargs)
        return synthetic_data

    def _fit_internal(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        """Internal fitting logic - implement your algorithm here"""
        pass

    def _generate_synthetic_data(self, n_samples: int):
        """Generate synthetic data during training"""
        # Implement synthetic data generation
        pass

    def _save_synthetic_data(self, x_synth: np.ndarray, y_synth: np.ndarray, synthetic_dir: str):
        """Save synthetic data to specified directory"""
        if synthetic_dir:
            import os
            os.makedirs(synthetic_dir, exist_ok=True)
            pd.DataFrame(x_synth).to_csv(f"{synthetic_dir}/x_synth.csv", index=False)
            pd.DataFrame(y_synth).to_csv(f"{synthetic_dir}/y_synth.csv", index=False)
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

### Step 6: Add Model-Specific Dependencies (if needed)

If your model requires specific dependencies, create:

`katabatic/models/your_model_name/pyproject.toml`:

```toml
[tool.poetry]
name = "katabatic-your-model"
version = "0.1.0"
description = "Your model implementation for Katabatic"

[tool.poetry.dependencies]
python = ">=3.11,<3.12"
# Add your model-specific dependencies here
tensorflow = "^2.19.0"  # Example
torch = "^2.0.0"        # Example

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Step 7: Test Your Model

Create a test script:

```python
from katabatic.models.your_model_name.models import YourModelName
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

# Test with existing pipeline
pipeline = TrainTestSplitPipeline(model=YourModelName)
pipeline.run(
    input_csv='discretized_data/car.csv',
    output_dir='sample_data/car',
    synthetic_dir='synthetic/car/your_model',
    real_test_dir='sample_data/car'
)
```

## 🔄 Adding New Pipelines

### Step 1: Create Pipeline Directory

```bash
mkdir -p katabatic/pipeline/your_pipeline_name
touch katabatic/pipeline/your_pipeline_name/__init__.py
touch katabatic/pipeline/your_pipeline_name/pipeline.py
```

### Step 2: Implement Pipeline Class

`katabatic/pipeline/your_pipeline_name/pipeline.py`:

```python
from katabatic.pipeline.base_pipeline import Pipeline
from katabatic.models.base_model import Model
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.utils.split_dataset import split_dataset
from typing import List, Type, Optional


class YourPipelineName(Pipeline):
    """
    Description of your pipeline.

    This pipeline implements [describe the workflow].

    Parameters
    ----------
    model : Model
        The model class to use for training
    evaluations : List[Type], optional
        List of evaluation classes to run
    override_evaluations : bool, default=False
        Whether to override default evaluations
    """

    # Default evaluations for this pipeline
    _evaluations = [TSTREvaluation]

    def __init__(self,
                 model: Model,
                 evaluations: Optional[List[Type]] = None,
                 override_evaluations: bool = False):
        super().__init__(model)

        if evaluations and override_evaluations:
            self._evaluations = evaluations
        elif evaluations:
            self._evaluations.extend(evaluations)

    def run(self, *args, **kwargs):
        """
        Run your pipeline with the given arguments.

        Parameters
        ----------
        input_csv : str
            Path to input CSV file
        output_dir : str
            Directory to save processed data
        *args, **kwargs
            Additional pipeline-specific parameters

        Returns
        -------
        str
            Success message or results
        """
        # Validate required parameters
        input_csv = kwargs.pop('input_csv', None)
        output_dir = kwargs.pop('output_dir', None)

        if not input_csv or not output_dir:
            raise ValueError("Both 'input_csv' and 'output_dir' must be provided.")

        # Step 1: Initialize model
        current_model = self.model()

        # Step 2: Implement your pipeline logic
        # Example: Data preparation
        self._prepare_data(input_csv, output_dir, *args, **kwargs)

        # Step 3: Train model
        current_model.train(output_dir, *args, **kwargs)

        # Step 4: Run evaluations
        for evaluation_class in self._evaluations:
            eval_instance = evaluation_class(*args, **kwargs)
            eval_instance.evaluate()

        return "Your pipeline executed successfully."

    def _prepare_data(self, input_csv: str, output_dir: str, *args, **kwargs):
        """Implement your data preparation logic here"""
        # Example implementation
        split_dataset(input_csv, output_dir, *args, **kwargs)

        # Add any additional preprocessing steps specific to your pipeline
        pass

    def __repr__(self):
        return f"YourPipelineName(model={self.model})"
```

### Step 3: Update Pipeline **init**.py

`katabatic/pipeline/your_pipeline_name/__init__.py`:

```python
from .pipeline import YourPipelineName

__all__ = ['YourPipelineName']
```

### Step 4: Test Your Pipeline

```python
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.your_pipeline_name.pipeline import YourPipelineName

pipeline = YourPipelineName(model=GANBLR)
result = pipeline.run(
    input_csv='discretized_data/car.csv',
    output_dir='sample_data/car',
    synthetic_dir='synthetic/car/ganblr',
    real_test_dir='sample_data/car'
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

    This evaluation implements [describe the evaluation approach].

    Parameters
    ----------
    synthetic_dir : str
        Directory containing synthetic data
    real_test_dir : str
        Directory containing real test data
    **kwargs
        Additional evaluation parameters
    """

    def __init__(self, synthetic_dir: str, real_test_dir: str, **kwargs):
        # Initialize base class
        super().__init__(model=None, dataset=None, **kwargs)

        self.synthetic_dir = synthetic_dir
        self.real_test_dir = real_test_dir
        self.kwargs = kwargs

        # Load data
        self.x_synth, self.y_synth, self.x_test, self.y_test = self._load_data()

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
        # Example: Multiple model evaluation
        models = self._get_evaluation_models()

        for model_name, model in models.items():
            metrics = self._evaluate_single_model(model, model_name)
            results[model_name] = metrics

        # Save results
        self._save_results(results)

        # Print results
        self._print_results(results)

        return results

    def _load_data(self):
        """Load synthetic and real test data"""
        x_synth = pd.read_csv(os.path.join(self.synthetic_dir, "x_synth.csv"))
        y_synth = pd.read_csv(os.path.join(self.synthetic_dir, "y_synth.csv")).values.ravel()
        x_test = pd.read_csv(os.path.join(self.real_test_dir, "x_test.csv"))
        y_test = pd.read_csv(os.path.join(self.real_test_dir, "y_test.csv")).values.ravel()

        return x_synth, y_synth, x_test, y_test

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
        # Train on synthetic data
        model.fit(self.x_synth, self.y_synth)

        # Predict on real test data
        y_pred = model.predict(self.x_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted')
        }

        return metrics

    def _save_results(self, results: Dict[str, Any]):
        """Save results to CSV file"""
        # Extract dataset and model names from paths
        parts = os.path.normpath(self.synthetic_dir).split(os.sep)
        model_name = parts[-1] if len(parts) > 0 else "unknown"
        dataset_name = parts[-2] if len(parts) > 1 else "unknown"

        # Create results directory
        results_dir = os.path.join("Results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        # Save to CSV
        output_path = os.path.join(results_dir, f"{model_name}_your_evaluation.csv")

        rows = []
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                rows.append([model_name, metric_name, round(value, 4)])

        df_results = pd.DataFrame(rows, columns=["Model", "Metric", "Value"])
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
from katabatic.evaluate.your_evaluation_name.evaluation import YourEvaluationName

# Test evaluation
evaluation = YourEvaluationName(
    synthetic_dir='synthetic/car/ganblr',
    real_test_dir='sample_data/car'
)

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

**Advanced Testing with Python Script:**

```bash
# Install test dependencies
poetry install --group dev

# Use the test runner script
python run_tests.py all                    # All tests
python run_tests.py unit --coverage        # Unit tests with coverage
python run_tests.py integration --verbose  # Integration tests (verbose)
python run_tests.py fast                   # Fast tests only
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
from katabatic.models.your_model import YourModel

@pytest.mark.unit
@pytest.mark.models
class TestYourModel:
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = YourModel()
        assert isinstance(model, YourModel)
        assert model.is_fitted is False

    def test_model_training(self, sample_dataset_files):
        """Test model training with mock data."""
        model = YourModel()

        # Use test fixtures for reliable data
        result = model.train(
            dataset=sample_dataset_files['dir'],
            epochs=5
        )

        assert "successful" in result.lower()
        assert model.is_fitted is True
```

**Writing Integration Tests:**

```python
@pytest.mark.integration
@pytest.mark.pipeline
class TestYourModelIntegration:
    def test_model_with_pipeline(self, temp_dir, sample_binary_dataset):
        """Test model integration with pipeline."""
        X, y = sample_binary_dataset

        # Create test files
        test_data = pd.concat([X, y], axis=1)
        input_csv = os.path.join(temp_dir, 'test_data.csv')
        test_data.to_csv(input_csv, index=False)

        # Test complete workflow
        model = YourModel()
        pipeline = TrainTestSplitPipeline(model)

        result = pipeline.run(
            input_csv=input_csv,
            output_dir=temp_dir,
            synthetic_dir=os.path.join(temp_dir, 'synthetic'),
            real_test_dir=temp_dir
        )

        assert "success" in result.lower()
        assert os.path.exists(os.path.join(temp_dir, 'synthetic', 'x_synth.csv'))
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
3. Test train/evaluate/sample method interfaces
4. Test integration with pipeline
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
4. Test integration with pipeline
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
import tempfile
import os
from katabatic.models.your_model_name.models import YourModelName
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


def test_new_model_integration():
    """Test new model with existing pipeline"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup test data
        test_csv = os.path.join(temp_dir, "test_data.csv")
        # Create test data...

        # Test pipeline with new model
        pipeline = TrainTestSplitPipeline(model=YourModelName)
        result = pipeline.run(
            input_csv=test_csv,
            output_dir=os.path.join(temp_dir, "output"),
            synthetic_dir=os.path.join(temp_dir, "synthetic"),
            real_test_dir=os.path.join(temp_dir, "output")
        )

        assert "successfully" in result
```

## 📖 Usage Examples

### Example 1: Basic Model Usage

```python
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from utils import discretize_preprocess

# Step 1: Preprocess raw data
dataset_path = "raw_data/car.csv"
output_path = "discretized_data/car.csv"
discretize_preprocess(dataset_path, output_path)

# Step 2: Run complete pipeline
input_csv = 'discretized_data/car.csv'
output_dir = 'sample_data/car'

pipeline = TrainTestSplitPipeline(model=GANBLR)
result = pipeline.run(
    input_csv=input_csv,
    output_dir=output_dir,
    synthetic_dir='synthetic/car/ganblr',
    real_test_dir='sample_data/car'
)

print(result)
```

### Example 2: Custom Evaluation Pipeline

```python
from katabatic.models.great.models import GReaT
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.evaluate.your_evaluation_name.evaluation import YourEvaluationName

# Create pipeline with custom evaluation
pipeline = TrainTestSplitPipeline(
    model=GReaT,
    evaluations=[YourEvaluationName],
    override_evaluations=True  # Use only custom evaluation
)

result = pipeline.run(
    input_csv='discretized_data/adult.csv',
    output_dir='sample_data/adult',
    synthetic_dir='synthetic/adult/great',
    real_test_dir='sample_data/adult'
)
```

### Example 3: Standalone Model Usage

```python
import pandas as pd
from katabatic.models.ganblr.models import GANBLR

# Load preprocessed data
X = pd.read_csv("sample_data/car/x_train.csv")
y = pd.read_csv("sample_data/car/y_train.csv").values.ravel()

# Train model directly
model = GANBLR()
model.fit(X, y, k=2, epochs=50, batch_size=32)

# Generate synthetic data
synthetic_data = model.sample(size=1000)
print(f"Generated {len(synthetic_data)} synthetic samples")

# Evaluate model
X_test = pd.read_csv("sample_data/car/x_test.csv")
y_test = pd.read_csv("sample_data/car/y_test.csv").values.ravel()
accuracy = model.evaluate(X_test, y_test, model='rf')
print(f"TSTR Accuracy: {accuracy:.4f}")
```

### Example 4: Advanced Configuration

```python
from katabatic.models.great.models import GReaT
from katabatic.pipeline.cross_validation.pipeline import CrossValidationPipeline

# Configure GReaT model with custom parameters
class CustomGReaT(GReaT):
    def __init__(self):
        super().__init__(
            llm='microsoft/DialoGPT-medium',
            epochs=200,
            batch_size=16,
            efficient_finetuning='lora'
        )

# Use with cross-validation pipeline
cv_pipeline = CrossValidationPipeline(
    model=CustomGReaT,
    n_splits=5
)

# Run cross-validation
results = cv_pipeline.run(
    input_csv='discretized_data/magic.csv',
    output_dir='cv_results/magic'
)
```

### Example 5: Jupyter Notebook Workflow

```python
# Cell 1: Setup and imports
from katabatic.models.ganblr.models import GANBLR
from katabatic.models.great.models import GReaT
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from utils import discretize_preprocess
import pandas as pd
import matplotlib.pyplot as plt

# Cell 2: Data preprocessing
dataset_path = "raw_data/nursery.csv"
output_path = "discretized_data/nursery.csv"
discretize_preprocess(dataset_path, output_path)

# Preview data
df = pd.read_csv(output_path)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df.iloc[:, -1].value_counts()}")

# Cell 3: Run GANBLR
pipeline_ganblr = TrainTestSplitPipeline(model=GANBLR)
result_ganblr = pipeline_ganblr.run(
    input_csv=output_path,
    output_dir='sample_data/nursery',
    synthetic_dir='synthetic/nursery/ganblr',
    real_test_dir='sample_data/nursery'
)

# Cell 4: Run GReaT
pipeline_great = TrainTestSplitPipeline(model=GReaT)
result_great = pipeline_great.run(
    input_csv=output_path,
    output_dir='sample_data/nursery',
    synthetic_dir='synthetic/nursery/great',
    real_test_dir='sample_data/nursery'
)

# Cell 5: Compare results
ganblr_results = pd.read_csv('Results/nursery/ganblr_tstr.csv')
great_results = pd.read_csv('Results/nursery/great_tstr.csv')

# Plot comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# GANBLR results
ganblr_acc = ganblr_results[ganblr_results['Metric'] == 'Accuracy']
ax[0].bar(ganblr_acc['Model'], ganblr_acc['Value'])
ax[0].set_title('GANBLR - TSTR Accuracy')
ax[0].set_ylabel('Accuracy')

# GReaT results
great_acc = great_results[great_results['Metric'] == 'Accuracy']
ax[1].bar(great_acc['Model'], great_acc['Value'])
ax[1].set_title('GReaT - TSTR Accuracy')
ax[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()
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
# Install missing model-specific dependencies
cd katabatic/models/your_model_name
poetry install
```

#### 3. Data Format Issues

```python
# Ensure proper data types
y = pd.read_csv("y_train.csv").values.ravel()  # Convert to 1D array
X = pd.read_csv("x_train.csv")  # Keep as DataFrame
```

#### 4. Path Issues

```python
# Use absolute paths or os.path.join
import os
synthetic_dir = os.path.join('synthetic', 'car', 'ganblr')
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
   assert os.path.exists(input_csv), f"File not found: {input_csv}"
   ```

### Getting Help

1. **Check Error Messages**: Read the full stack trace
2. **Review Documentation**: Ensure you're following the correct API
3. **Check Examples**: Compare with working examples in `example.ipynb`
4. **Create Minimal Reproduction**: Isolate the issue with minimal code

## 🚀 Advanced Development Topics

### Adding Model-Specific Optimizations

For models with special requirements:

```python
class OptimizedModel(Model):
    def train(self, dataset: str, size_category: str = 'small', **kwargs):
        # Adjust parameters based on dataset size
        if size_category == 'large':
            self.batch_size = 128
            self.epochs = 50
        elif size_category == 'small':
            self.batch_size = 32
            self.epochs = 100

        # Continue with training...
```

### Custom Data Loaders

For specialized data handling:

```python
class CustomDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_with_preprocessing(self):
        # Custom loading logic
        return x_train, y_train, x_test, y_test
```

### Experiment Tracking Integration

```python
import wandb

class TrackedModel(Model):
    def train(self, *args, **kwargs):
        wandb.init(project="katabatic-experiments")
        # Log parameters and metrics
        wandb.log({"epoch": epoch, "loss": loss})
```

---

This development guide should serve as your comprehensive reference for contributing to the Katabatic framework. For questions or clarifications, please reach out to the development team or create an issue in the repository.
