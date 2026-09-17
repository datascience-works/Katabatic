# TableGAN

This module integrates a TableGAN-style synthetic tabular data generator into Katabatic.

The implementation converts each tabular record into a square numeric matrix and trains convolutional generator and discriminator networks over that representation. It also includes a feature-statistics information loss and an auxiliary classifier objective intended to help preserve relationships between the feature columns and the target column.

## Status

Experimental.

The model is registered in Katabatic with:

```text
supported = False
```

TableGAN is therefore available as an experimental model but is not currently part of Katabatic's officially supported model set.

## Dependencies

Install the TableGAN optional dependencies with:

```bash
pip install "katabatic[tablegan]"
```

The TableGAN integration requires PyTorch.

## Usage

```python
from katabatic.models.tablegan import TableGANModel

model = TableGANModel(
    epochs=100,
    batch_size=64,
    seed=42,
)

model.train(
    data_dir="path/to/dataset",
    categorical_cols=["category", "target"],
    continuous_cols=["age", "income"],
)

synthetic = model.sample(1000)
```

The model can also be created through the Katabatic model registry:

```python
from katabatic.models.registry import ModelRegistry

model = ModelRegistry.create_model(
    "tablegan",
    epochs=100,
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

When `x_train.csv` and `y_train.csv` are used, `y_train.csv` must contain exactly one target column. The feature and target data are combined internally with the target as the final column.

The current TableGAN integration uses the final column of the training DataFrame as the target for the auxiliary classifier and for `y_synth.csv`.

## Parameters

The main configuration parameters are:

- `epochs`: number of training epochs. Default: `100`.
- `batch_size`: training and sampling batch size. Default: `64`.
- `noise_dim`: dimension of the generator noise vector. Default: `100`.
- `base_channels`: base number of convolutional channels. Default: `64`.
- `information_weight`: weight applied to the information loss. Default: `1.0`.
- `delta_mean`: tolerance applied to the difference between real and generated feature means. Default: `0.0`.
- `delta_std`: tolerance applied to the difference between real and generated feature standard deviations. Default: `0.0`.
- `classifier_weight`: weight applied to the auxiliary classifier loss. Default: `1.0`.
- `lr`: Adam optimizer learning rate. Default: `0.0002`.
- `betas`: Adam optimizer beta values. Default: `(0.5, 0.999)`.
- `seed`: random seed used during training. Default: `42`.
- `device`: PyTorch device used for training. Default: `"cpu"`.

## Data Representation

Before training, each column is converted to a numeric value in the range `[-1, 1]`.

Continuous columns are min-max scaled using their observed training minimum and maximum.

Categorical columns are represented by category indices that are scaled to `[-1, 1]`. The fitted category values are retained so generated values can be mapped back to known categories during sampling.

Each encoded row is padded with zeros and reshaped into a square single-channel matrix. The matrix side length is a power of two with a minimum size of 4 so that it can be processed by the convolutional networks.

## Training

Training consists of three interacting networks:

1. A generator produces synthetic table matrices from random noise.
2. A discriminator learns to distinguish real matrices from generated matrices.
3. An auxiliary classifier learns to predict the encoded final-column target from the remaining cells in each matrix.

The generator is trained using a combination of:

- adversarial loss from the discriminator,
- information loss based on differences between real and generated feature statistics, and
- auxiliary classification loss.

After successful training, `model.is_fitted` is set to `True`.

## Sampling

Synthetic records can be generated after training:

```python
synthetic = model.sample(1000)
```

This returns a pandas DataFrame with the same column names and ordering as the training data.

If `n` is not supplied, the current implementation generates 1000 rows.

Calling `sample()` before training raises a `RuntimeError`.

## Output

Unless another output directory is supplied, training saves synthetic output under:

```text
synthetic/<dataset>/tablegan/
```

The generated files are:

```text
x_synth.csv
y_synth.csv
metadata.json
```

`x_synth.csv` contains the generated feature columns.

`y_synth.csv` contains the generated final target column.

`metadata.json` records the fitted schema, target column, original data types, categorical and continuous columns, training configuration, and matrix side length.

A custom output directory can be supplied using:

```python
model.train(
    data_dir="path/to/dataset",
    synthetic_dir="path/to/output",
)
```

## Artifact Persistence

The fitted TableGAN state can be persisted as:

```text
tablegan_state.pt
```

The artifact contains the fitted transformation metadata and the state dictionaries for the generator, discriminator, and auxiliary classifier.

`TableGANModel.load_from_ref()` reconstructs the networks and restores the saved model state through Katabatic's artifact interface.

## Evaluation

TableGAN implements Katabatic's standard model interface:

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

`TableGANModel` inherits from Katabatic's base `Model` class and implements:

```text
train()
sample()
evaluate()
get_required_dependencies()
load_from_ref()
```

The required Python dependency reported by the model is:

```python
["torch"]
```

## Limitations

The current TableGAN integration has the following limitations:

- TableGAN is experimental in Katabatic.
- The implementation is adapted to Katabatic's dataset, model, and artifact interfaces and should not be treated as an exact reproduction of every detail of the original TableGAN implementation.
- The final column is treated as the target for the auxiliary classifier.
- Categorical columns are represented using scalar category positions, which introduces an ordering into the internal representation.
- Decoded categorical values are returned using the stored string category representation rather than restoring every original pandas categorical dtype exactly.
- Missing-value handling is not currently implemented explicitly.
- `evaluate()` currently returns the pipeline-compatible placeholder value of `0.0`.
- Model quality and performance depend on the dataset and training configuration; benchmark results have not yet been documented for this integration.

## References

TableGAN was introduced as a GAN-based approach for synthesizing tabular data using a convolutional representation together with additional objectives intended to improve the statistical and semantic properties of generated records.

This Katabatic module provides an adapted implementation for Katabatic's model interface, dataset conventions, synthetic output structure, artifact system, model registry, and optional dependency system.