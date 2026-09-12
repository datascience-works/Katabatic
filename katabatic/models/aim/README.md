# Adaptive Iterative Mechanism (AIM)

This directory contains an experimental implementation of the Adaptive
Iterative Mechanism (AIM) for synthetic tabular data generation in Katabatic.

AIM is a differentially private synthetic data method based on iteratively
selecting and measuring informative marginal queries before fitting a graphical
model to the noisy measurements.

## Katabatic integration

The Katabatic implementation provides an `AIMModel` class that follows the
standard Katabatic model interface.

The model supports:

- training from Katabatic benchmark split directories;
- categorical and continuous column metadata;
- discretisation of continuous variables;
- differential privacy through `epsilon` and `delta`;
- synthetic data generation through `sample`;
- Katabatic-compatible synthetic output files;
- integration with the model registry and benchmark workflow.

## Dependencies

AIM uses the `private-pgm` package for graphical-model based inference.

Install the optional AIM dependencies with:

    pip install "katabatic[aim]"

The dependency is pinned to a specific `private-pgm` commit for reproducibility.

## Basic usage

    from katabatic.models.aim import AIMModel

    model = AIMModel(
        epsilon=1.0,
        delta=1e-5,
        degree=2,
        max_bins=10,
        seed=42,
    )

    model.train(
        data_dir="path/to/splits",
        synthetic_dir="path/to/synthetic",
        categorical_cols=["category", "target"],
        continuous_cols=["age", "income"],
    )

    synthetic_df = model.sample(1000)

The training directory may contain `train_full.csv`. If it is unavailable,
the model can load Katabatic-style `x_train.csv` and `y_train.csv` files.

## Continuous columns

AIM operates on discrete domains. Continuous columns are therefore converted
into quantile-based bins before training.

Synthetic values are decoded using representative values from the original
training data. This allows the generated output to retain the original column
structure while keeping the AIM inference domain discrete.

## Main parameters

- `epsilon`: Differential privacy epsilon value.
- `delta`: Differential privacy delta value.
- `degree`: Maximum marginal interaction degree used to build the workload.
- `max_bins`: Maximum number of quantile bins used for continuous columns.
- `max_cells`: Maximum number of cells allowed for a workload marginal.
- `max_model_size`: Maximum graphical-model size used during AIM inference.
- `max_iters`: Maximum number of inference iterations.
- `rounds`: Number of AIM adaptive measurement rounds.
- `seed`: Random seed for reproducibility.

## Benchmark compatibility

`AIMModel.train` accepts Katabatic benchmark column metadata using
`categorical_cols` and `continuous_cols`.

After training, `sample(n)` returns a pandas DataFrame aligned with the training
columns, allowing AIM to be used by Katabatic's generic benchmark runner.

## Status

AIM is currently registered as an experimental model and is not listed as an
officially supported Katabatic model.

## Attribution

The AIM mechanism is adapted from the AIM implementation in Ryan McKenna's
`private-pgm` research code.

The concentrated differential privacy conversion routines are adapted from
IBM's discrete Gaussian differential privacy work.

The relevant upstream code is distributed under the Apache License 2.0.
