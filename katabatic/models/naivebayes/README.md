# Naive Bayes — Class-Conditional Generator

Naive Bayes is a simple and efficient baseline for generating synthetic tabular data. It uses the **Naive Bayes conditional-independence assumption**, modelling features independently given the target class.

The model:

1. Learns the distribution of each feature for each target class.
2. Samples a target class based on the training data.
3. Generates each feature conditionally on that class.
4. Combines the generated values into synthetic records.

It supports both **categorical and continuous features**. Categorical features use class-conditional multinomial distributions with Laplace smoothing, while continuous features use Gaussian distributions.

## Project Structure

* `models.py` — Contains `NaiveBayesModel` and the `train()`, `evaluate()`, and `sample()` functionality.
* `utils.py` — Helper functions for feature-type detection and feature processing.
* `__init__.py` — Exports the Naive Bayes model.

## Usage

```python
from katabatic.models.naivebayes.models import NaiveBayesModel

model = NaiveBayesModel(seed=42)

model.train(
    data_dir="benchmarks/splits/car",
    synthetic_dir="benchmarks/synthetic/car/naivebayes",
    categorical_cols=[
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
    ],
    continuous_cols=[],
)

synthetic_df = model.sample(n=1000)
```

A random seed can be provided to make synthetic data generation reproducible.

## Benchmark Results

The model was tested using the Katabatic benchmark pipeline on **Adult, Magic, Nursery, and Shuttle** datasets.

| Dataset |        Composite Score |
| ------- | ---------------------: |
| Magic   |                 0.8620 |
| Nursery |                 0.8643 |
| Shuttle |             **0.9094** |
| Adult   | Successfully evaluated |

Shuttle produced the strongest reported result, with a composite score of **0.9094**.

## Strengths

* Simple and fast implementation
* Low computational requirements
* No GPU required
* Supports categorical and continuous features
* Reproducible generation using random seeds
* Integrates with the Katabatic `Model` interface
* Useful as a baseline for comparing more complex models

## Limitations

The main limitation is the **conditional-independence assumption**. Because features are treated as independent once the target class is known, the model does not directly learn complex relationships between features.

This can lead to:

* Reduced preservation of feature correlations
* Unrealistic combinations of feature values
* Lower consistency
* Synthetic data that is easier to distinguish from real data

The Nursery benchmark also showed a high number of exact duplicates, resulting in a lower privacy score.

## Dependencies

* `numpy`
* `pandas`
* `scikit-learn`

The model runs within the existing Katabatic Poetry environment.

## Conclusion

Naive Bayes provides a **fast, lightweight baseline** for class-conditional synthetic tabular data generation. It performs well on several benchmark dimensions and integrates directly with the Katabatic framework.

However, its simplicity limits its ability to reproduce complex feature relationships, making it best suited as a **baseline or comparison model** rather than a replacement for more advanced generative approaches.
