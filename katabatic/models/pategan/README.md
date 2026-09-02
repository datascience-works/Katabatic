# PATE-GAN

**PATE-GAN (Private Aggregation of Teacher Ensembles - Generative Adversarial Network)** is a differentially private synthetic data generation model introduced by Jordon, Yoon, and van der Schaar at ICLR 2019.

This Katabatic implementation is aligned primarily with the original authors' released PATE-GAN source implementation, while the research paper is used to clarify the intended PATE framework and teacher data partitioning.

## References

**Paper**

Jordon, J., Yoon, J., & van der Schaar, M. (2019).
*PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees.*
International Conference on Learning Representations (ICLR 2019).

**Original authors' implementation**

`vanderschaarlab/mlforhealthlabpub/alg/pategan/pate_gan.py`

## Overview

PATE-GAN combines a generative model with the Private Aggregation of Teacher Ensembles (PATE) framework.

The training procedure consists of:

1. Splitting the private training dataset into disjoint teacher partitions.
2. Training a separate teacher classifier on each partition.
3. Generating candidate synthetic samples with the generator.
4. Obtaining predictions from all teachers for generated samples.
5. Privately aggregating teacher predictions using Laplace noise.
6. Training a student discriminator using the private teacher labels.
7. Updating the generator against the student discriminator.
8. Tracking cumulative privacy loss using a moments accountant.
9. Stopping when the privacy budget is reached or Katabatic's iteration safety cap is reached.

The generator produces the complete transformed data row, including features and the target column.

## Implementation Basis

Katabatic follows the original authors' released implementation for the core executable model.

### Generator

The source-aligned generator uses:

- latent dimension equal to the transformed input dimension
- latent samples drawn from `Uniform(-1, 1)`
- two hidden layers
- hidden width of `4 * input_dimension`
- `tanh` hidden activations
- `sigmoid` output activation
- Xavier-style weight initialization

### Student Discriminator

The student uses:

- one hidden layer
- hidden width equal to the input dimension
- ReLU activation
- linear output
- RMSProp optimization
- weight clipping to `[-0.01, 0.01]`

The student is trained only on generated samples labelled using the private teacher ensemble.

### Teacher Ensemble

The private training data is divided into disjoint partitions.

Each teacher is trained only on its corresponding partition using a logistic regression classifier.

Katabatic intentionally uses the teacher's corresponding partition rather than reproducing a stale-loop-index issue present in the released source. This follows the PATE-GAN paper's intended design in which teacher `i` is trained only on partition `D_i`.

### Private Teacher Aggregation

Teacher predictions are aggregated using the mechanism from the authors' released source implementation.

For each generated sample:

1. Each teacher predicts class `0` or `1`.
2. The number of class-0 and class-1 votes is counted.
3. Laplace noise is added during aggregation.
4. The noisy vote result is converted into the private student label.

The `lamda` parameter controls the Laplace mechanism used by the released implementation.

### Privacy Accounting

PATE-GAN maintains a moments accountant during training.

The accumulated privacy cost is used to estimate `epsilon_hat` for the configured `delta`.

Training continues while:

```text
epsilon_hat < epsilon
```

Katabatic additionally applies `niter` as a maximum-iteration safety cap. This prevents an unexpectedly long run if the requested privacy stopping condition is not reached within a practical number of iterations.

## Paper and Released Source Differences

The ICLR 2019 paper and the authors' released implementation are not identical in every implementation detail.

The paper's experimental appendix describes:

- generator and student discriminator depth of 3
- teacher discriminator depth of 1
- hidden layer sizes based on `d`, `d/2`, and `d`
- ReLU activations except for a sigmoid output
- batch size of 64
- `nT = nS = 5`
- learning rate of `1e-4`
- Adam optimizer

The released source implementation instead uses:

- a generator with two `tanh` hidden layers of width `4d`
- a student with one ReLU hidden layer of width `d`
- logistic regression teacher models
- RMSProp for generator and student optimization
- student weight clipping
- latent samples from `Uniform(-1, 1)`

The paper also describes noisy PATE aggregation in terms of adding independent Laplace noise to class vote counts, whereas the released source performs its binary aggregation using a single Laplace noise draw in the class-vote calculation.

Katabatic follows the released source behavior for these executable implementation details while documenting the differences from the paper.

## Configuration

### Core Parameters

- **epsilon** (`float`, default: `1.0`)
  Target privacy budget used by the moments accountant stopping condition.

- **delta** (`float`, default: `1e-5`)
  Delta value used when calculating the accumulated privacy loss.

- **num_teachers** (`int`, default: `10`)
  Number of teacher models and disjoint private-data partitions.

- **batch_size** (`int`, default: `64`)
  Number of generated samples used during student/generator training. The Katabatic benchmark scripts use 64, consistent with the paper's reported experimental batch size.

- **learning_rate** (`float`, default: `1e-4`)
  Learning rate used by the RMSProp optimizers in the source-aligned implementation.

- **lamda** (`float`, default: `1.0`)
  Laplace mechanism parameter used during private teacher aggregation.

- **n_s** (`int`, default: `5`)
  Number of student updates performed during each outer training iteration.

- **z_dim** (`int`, optional)
  Latent dimension. When not supplied, it is set equal to the transformed input dimension, following the released source implementation.

- **niter** (`int`, default: `10000`)
  Katabatic maximum-iteration safety cap.

- **random_state** (`int`, default: `42`)
  Random seed used for reproducibility.

## Quick Start

```python
import pandas as pd

from katabatic.models.pategan import PATEGAN

X_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")

model = PATEGAN(
    epsilon=1.0,
    delta=1e-5,
    num_teachers=10,
    niter=10000,
    batch_size=64,
    learning_rate=1e-4,
    lamda=1.0,
    n_s=5,
    random_state=42,
)

model.fit(X_train, y_train, verbose=1)

synthetic_data = model.sample(n=1000)
```

## Katabatic Integration

PATE-GAN uses Katabatic's existing tabular preprocessing and evaluation workflow rather than reproducing the paper-specific evaluation pipeline.

The transformer handles categorical and continuous columns and converts the training data into the numerical representation required by PATE-GAN. Generated samples are inverse-transformed back into the original tabular representation.

Katabatic continues to evaluate PATE-GAN using its standard datasets and evaluation metrics so that its results remain comparable with the other synthetic-data models in the repository.

This integration layer does not change the core PATE-GAN training objective for the purpose of model optimization.

## Benchmark Scripts

Katabatic provides PATE-GAN benchmark scripts for:

- Adult
- Bank Marketing
- Car Evaluation
- Covertype
- Credit Card

The benchmark scripts use a batch size of `64` following the experimental configuration reported in the paper.

Dataset-specific preprocessing remains part of the Katabatic benchmark pipeline.

## Model Contract

### Inputs

The standard Katabatic training workflow uses:

```text
dataset_dir/x_train.csv
dataset_dir/y_train.csv
```

The target data is combined with the feature data before transformation because PATE-GAN generates complete synthetic rows.

### Outputs

The Katabatic workflow produces:

```text
synthetic_dir/x_synth.csv
synthetic_dir/y_synth.csv
synthetic_dir/metadata.json
```

The metadata records the model configuration and relevant source-alignment information.

## Alignment Notes

This implementation intentionally avoids optimization-specific modifications that are not part of the original PATE-GAN design.

In particular, the aligned implementation does not use:

- WGAN-GP or gradient penalty
- Gaussian teacher-vote noise
- class-conditional/class-aware generation
- independently optimized neural teacher architectures
- dummy target-class injection
- dataset-specific hyperparameter tuning as part of the core model

The goal of this implementation is faithful reproduction and integration of the original PATE-GAN source behavior within Katabatic rather than maximizing benchmark performance.

## Validation

PATE-GAN should be validated through Katabatic's standard evaluation pipeline after implementation alignment.

Validation results should be interpreted separately from the results reported in the original PATE-GAN paper because Katabatic uses its own datasets, preprocessing, train/test splits, and evaluation framework.

Therefore, differences between Katabatic scores and the paper's reported results do not by themselves indicate an implementation mismatch.
