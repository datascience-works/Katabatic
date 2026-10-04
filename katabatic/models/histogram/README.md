# Histogram Model

## Overview

The Histogram model is a simple statistical method for generating synthetic tabular data.

The model learns the empirical distribution of each column from the real dataset using value frequencies. It then generates synthetic data by sampling values from these learned distributions.

Each column is sampled independently. Therefore, the model can preserve the individual distribution of each column, but relationships and correlations between different columns may not be fully preserved.

## Model Files

The Histogram implementation follows the Katabatic three-file model structure:

- `__init__.py` – Defines how the `HistogramModel` is exposed and imported within Katabatic.
- `models.py` – Contains the main Histogram model implementation and generation logic.
- `utils.py` – Contains helper functions for calculating column distributions and sampling values.

## Status

Histogram is an officially supported and has no dependencies beyond Katabatic's core install.

```bash
pip install katabatic[histogram]   # or: poetry install -E histogram
```

## Usage

`train()` reads `train_full.csv` (or `x_train.csv` + `y_train.csv`) from a directory, fits the model, and writes `x_synth.csv` / `y_synth.csv`. Pass `artifact_state_dir` (the artifact pipeline does this automatically) to persist state for `HistogramModel.load_from_ref()`.

```python
from katabatic.models.histogram import HistogramModel

model = HistogramModel(random_state=42)

model.train("path/to/split_dir", synthetic_dir="path/to/synthetic_dir")

synthetic_data = model.sample(
    len(real_data),
    seed=42,
)
```

Through the artifact pipeline:

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

TrainTestSplitPipeline(model=HistogramModel()).run(
    input_csv="data.csv",
    dataset_name="mydata",
    artifact_store=store,
    model_name="histogram",
)
```

Benchmark scripts for all five datasets live in [benchmarks/examples/histogram/](../../../benchmarks/examples/histogram/).

## How It Works

During training, the Histogram model:

1. Calculates the frequency of each unique value in every column.
2. Converts these frequencies into probabilities.
3. Stores the unique values and their corresponding probabilities.

During synthetic data generation, the model independently samples values for each column according to the learned probabilities and combines them to create a synthetic dataset.

## Advantages

- Simple and easy to understand.
- Fast model training and synthetic data generation.
- Supports both numerical and categorical data.
- Generates the requested number of synthetic rows.
- Supports reproducible synthetic data generation using a random seed.

## Limitations

The Histogram model treats every column independently. As a result, it may not fully preserve correlations or complex relationships between columns in the original dataset.

Values are resampled from those observed in the training data (no binning or smoothing), so numeric columns never produce values outside the observed set. For evaluation, use the Katabatic evaluation pipeline.

## Evaluation

The Histogram model was evaluated using the Katabatic benchmarking framework on all five required datasets:

- Adult
- Shuttle
- Car
- Magic
- Nursery

The evaluation measures the following six dimensions:

- Fidelity
- Utility
- Diversity
- Privacy
- Consistency
- Stability

The model successfully completed the evaluation pipeline on all five datasets.

## Reference and Credit

This Histogram implementation was adapted from the Histogram model contributed by **Rishi Goyal** to the Katabatic mentorship repository.

Source implementation:

https://github.com/katabatic-mentorship/katabatic-mentorship-repo/tree/Rishi_Goyal

Original model location:

`katabatic/models/histogram_Rishi`

The implementation was adapted to follow the Katabatic model acceptance criteria, including the required three-code-file structure, runner scripts, reproducible sampling, and compatibility with the Katabatic evaluation and performance benchmarking framework.
