# Gaussian Copula

Gaussian Copula is a tabular generative model implemented in Katabatic using SDV's `GaussianCopulaSynthesizer`.

This model is useful for generating synthetic tabular data while preserving the structure of a dataset and learning column relationships through a Gaussian copula-based distribution.

## Overview

- Model class: `GaussianCopulaModel`
- Backend library: SDV
- Use case: synthetic tabular data generation
- Status: experimental in the Katabatic registry

The implementation in Katabatic wraps SDV and fits a single-table metadata model from the training dataframe, then samples synthetic rows using the fitted synthesizer.

## Installation

Install the SDV dependency:

```bash
pip install sdv
```

If you are using the project environment with Poetry, you can also install it in the active environment:

```bash
poetry add sdv
```

## Usage

### Basic model training

```python
from katabatic.models.gaussian_copula.models import GaussianCopulaModel

model = GaussianCopulaModel()
model.train(data_dir="sample_data/car")
```

### Training with split feature and target files

The model accepts either:

- `train_full.csv`, or
- `x_train.csv` + `y_train.csv`

Example folder structure:

```text
sample_data/car/
├── train_full.csv
└── test_full.csv
```

Or:

```text
sample_data/car/
├── x_train.csv
├── y_train.csv
├── x_test.csv
├── y_test.csv
```

### Synthetic output

By default, synthetic data is saved under:

```text
synthetic/<dataset_name>/gaussian_copula/
```

The output contains:

- `x_synth.csv`
- `y_synth.csv`

### Direct sampling

```python
from katabatic.models.gaussian_copula.models import GaussianCopulaModel

model = GaussianCopulaModel()
model.train(data_dir="sample_data/car")

synthetic = model.sample(n=100)
print(synthetic.head())
```

## Model behavior

During training, the model does the following:

1. Loads the training data from `train_full.csv` or split train files.
2. Detects metadata from the dataframe using SDV.
3. Builds a `GaussianCopulaSynthesizer`.
4. Fits the synthesizer to the training data.
5. Generates synthetic rows matching the count of the training table.
6. Saves the synthetic feature and target splits into a folder under `synthetic/`.

## Benchmark example

The repository includes benchmark scripts such as:

- `benchmarks/examples/gaussian_copula/car.py`
- `benchmarks/examples/gaussian_copula/adult.py`
- `benchmarks/examples/gaussian_copula/magic.py`
- `benchmarks/examples/gaussian_copula/nursery.py`
- `benchmarks/examples/gaussian_copula/shuttle.py`

These scripts use the Katabatic pipeline with the Gaussian Copula model and evaluate synthetic quality via the train-on-synthetic, test-on-real workflow.

## Notes

- This model depends on SDV and is not a core supported model in the same tier as GANBLR or GReaT.
- It is best suited for tabular datasets with a clear single target column when using the split `x_train.csv` / `y_train.csv` format.
- The final column is treated as the target column in the training pipeline.

## Related files

- [katabatic/models/gaussian_copula/models.py](models.py)
- [katabatic/models/gaussian_copula/utils.py](utils.py)
- [katabatic/models/registry.py](../registry.py)

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
