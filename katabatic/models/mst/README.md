# Maximum Spanning Tree (MST)

This module integrates the Maximum Spanning Tree (MST) synthetic data generator from SmartNoise Synth into Katabatic.

MST is a differentially private synthetic data generation method designed primarily for discrete and categorical tabular data. It learns relationships between attributes using a maximum spanning tree and generates synthetic records while respecting a specified differential privacy budget.

## Status

Experimental.

The model is registered in Katabatic with:

```text
supported = False
```

MST is therefore available as an experimental model but is not currently part of Katabatic's officially supported model set.

## Dependencies

Install the MST optional dependencies with:

```bash
pip install "katabatic[mst]"
```

The MST integration uses:

- SmartNoise Synth
- OpenDP
- Private-PGM (`mbi`)

The Private-PGM dependency is pinned to the revision required by the tested SmartNoise MST implementation.

## Usage

```python
from katabatic.models.mst import MSTModel

model = MSTModel(
    epsilon=3.0,
    categorical_columns=[
        "workclass",
        "education",
        "occupation",
        "class",
    ],
)

model.train(
    data_dir="path/to/dataset",
)

synthetic = model.sample(1000)
```

The model can also be created through the Katabatic model registry:

```python
from katabatic.models.registry import ModelRegistry

model = ModelRegistry.create_model(
    "mst",
    epsilon=3.0,
)
```

## Input Data

Katabatic training data can be provided using either:

```text
train_full.csv
```

or separate feature and target files:

```text
x_train.csv
y_train.csv
```

When `x_train.csv` and `y_train.csv` are used, `y_train.csv` must contain exactly one target column.

The target is expected to be the final column of the combined training DataFrame.

Training data containing missing values is currently rejected by the MST wrapper.

## Parameters

### `epsilon`

Controls the differential privacy budget.

A smaller epsilon provides stronger privacy but generally introduces more noise. A larger epsilon provides weaker privacy and may preserve more information from the original dataset.

Default:

```python
epsilon=3.0
```

The value must be greater than zero.

### `delta`

Optional approximate differential privacy parameter.

If `delta` is not provided, Katabatic calculates it using:

```text
1 / (n * sqrt(n))
```

where `n` is the number of rows in the training dataset.

When explicitly supplied, `delta` must be greater than zero and less than one.

### `categorical_columns`

Optional list containing the names of categorical columns.

For example:

```python
categorical_columns=[
    "workclass",
    "education",
    "occupation",
    "class",
]
```

If the list is not supplied, Katabatic automatically treats columns with object, category, or boolean data types as categorical.

MST is most naturally suited to categorical or discretised data. Continuous variables may require appropriate discretisation before training depending on the dataset and intended use.

## Training

Training is performed using:

```python
model.train(
    data_dir="path/to/dataset",
)
```

During training, the Katabatic wrapper:

1. Loads the training dataset.
2. Validates that the dataset is not empty and does not contain missing values.
3. Resolves the categorical columns.
4. Resolves the differential privacy `delta` value.
5. Creates the SmartNoise MST synthesizer.
6. Fits MST to the training data.
7. Generates a synthetic dataset with the same number of rows as the original training dataset.
8. Saves the generated features, target, and metadata.

After successful training:

```python
model.is_fitted
```

is set to `True`.

## Sampling

Additional synthetic records can be generated after training:

```python
synthetic = model.sample(1000)
```

This returns a pandas DataFrame containing 1000 generated rows.

If `n` is not supplied:

```python
synthetic = model.sample()
```

the model generates the same number of rows as the original training dataset.

Calling `sample()` before training raises a `RuntimeError`.

## Output

Unless another output directory is supplied, training saves synthetic output under:

```text
synthetic/<dataset>/mst/
```

The generated files are:

```text
x_synth.csv
y_synth.csv
metadata.json
```

### `x_synth.csv`

Contains the generated feature columns.

### `y_synth.csv`

Contains the generated target column.

### `metadata.json`

Records information about the training run, including:

- original schema
- target column
- original data types
- categorical columns
- epsilon
- delta
- number of original rows
- number of generated rows

A custom output directory can be supplied using:

```python
model.train(
    data_dir="path/to/dataset",
    synthetic_dir="path/to/output",
)
```

## Evaluation

MST implements Katabatic's standard model interface:

```python
model.evaluate()
```

The current implementation returns:

```text
0.0
```

as a placeholder evaluation score for compatibility with the existing Katabatic model pipeline.

Calling `evaluate()` before training raises a `RuntimeError`.

## Katabatic Model Interface

`MSTModel` inherits from Katabatic's base `Model` class and implements:

```text
train()
sample()
evaluate()
get_required_dependencies()
```

The required Python dependencies reported by the model are:

```python
[
    "snsynth",
    "mbi",
    "opendp",
]
```

## Differential Privacy

MST uses differential privacy to limit the influence that individual training records can have on the generated synthetic dataset.

The privacy behaviour is controlled primarily through `epsilon` and `delta`.

The appropriate privacy parameters depend on the intended application and privacy requirements. The default values provided by this wrapper are intended to provide a usable model configuration rather than define a universal privacy guarantee for every dataset or use case.

## OpenDP Compatibility

The tested integration uses SmartNoise Synth 1.0.5 together with OpenDP 0.14.x.

SmartNoise Synth 1.0.5 creates floating-point OpenDP domains in MST's privacy helper functions. With the OpenDP version used by this integration, those domains must explicitly disallow NaN values when used with `AbsoluteDistance`.

The Katabatic MST wrapper applies this compatibility adjustment at runtime by using OpenDP floating-point domains with:

```python
nan=False
```

The compatibility adjustment is limited to the running Python process. It does not modify the installed SmartNoise Synth, OpenDP, or Private-PGM source files.

## Limitations

The current MST integration has the following limitations:

- MST is experimental in Katabatic.
- It is primarily suited to categorical or discretised tabular data.
- Missing values are not currently accepted during training.
- Automatic categorical detection is based on pandas object, category, and boolean data types.
- Integer-encoded categorical columns should be supplied explicitly through `categorical_columns`.
- Model artifact persistence is not yet implemented.
- `evaluate()` currently returns the pipeline-compatible placeholder value of `0.0`.
- The integration currently includes an OpenDP compatibility adjustment for the tested SmartNoise Synth version.

## References

The MST implementation is provided through SmartNoise Synth.

SmartNoise Synth provides differentially private synthetic data generation methods and uses Private-PGM (`mbi`) for the graphical-model inference required by MST.

This Katabatic module provides the wrapper required to use MST through Katabatic's model interface, dataset conventions, synthetic output structure, model registry, and dependency system.
