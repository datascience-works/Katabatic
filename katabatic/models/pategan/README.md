# PATE-GAN

**PATE-GAN (Private Aggregation of Teacher Ensembles - Generative Adversarial Network)** is a
differentially private synthetic data generation approach introduced by Jordon, Yoon, and van der
Schaar at ICLR 2019.

Katabatic's implementation takes inspiration from the paper's core idea — inject differential
privacy noise into the discriminator's training signal, then train a generator against it via a
WGAN-style adversarial loss — but is a from-scratch, simplified adaptation rather than a literal
reproduction of the paper's teacher-ensemble architecture. See [Implementation Notes](#implementation-notes)
for exactly how it differs.

## Reference

Jordon, J., Yoon, J., & van der Schaar, M. (2019).
*PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees.*
International Conference on Learning Representations (ICLR 2019).

## Overview

1. The training data (features + target, concatenated into one frame) is encoded into a
   numeric representation by a `DataTransformer`.
2. A generator and discriminator are built as small feedforward networks (see
   [Architecture](#architecture)) and trained adversarially with a WGAN-GP loss
   (`katabatic/models/pategan/models.py::_build_model`).
3. Each outer iteration, the discriminator is updated `num_teachers` times on independently
   sampled batches; each update's real/fake labels have differential-privacy noise mixed in via
   `PrivacyMechanism.add_gaussian_noise` (a Gaussian mechanism, not the paper's Laplace
   aggregation) before being thresholded back to binary labels. The generator is then updated
   once against the current discriminator.
4. Training runs for a fixed `niter` iterations (there is no moments-accountant stopping
   condition or `epsilon_hat` tracking in this implementation — `epsilon`/`delta` configure the
   noise scale via `PrivacyMechanism`, not a runtime privacy budget check).

The generator produces the complete synthetic row (features and target together); the
transformer's inverse transform splits it back into the original tabular representation.

## Architecture

- **Generator**: 2 hidden layers, width `h_dim = input_dim`, `tanh` activations, `sigmoid`
  output. Latent samples drawn from `Uniform(-1, 1)` with `z_dim` defaulting to
  `max(input_dim // 4, 2)` when not set explicitly.
- **Discriminator**: 2 hidden layers, width `h_dim = input_dim`, ReLU activations, linear output.
- Both networks use the Adam optimizer (`beta1=0.5`); the discriminator loss includes a WGAN-GP
  gradient penalty term (coefficient `lambda_gp`), not weight clipping.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epsilon` | float | `1.0` | Privacy budget; scales the Gaussian noise added to teacher labels (lower = more noise = more private). |
| `delta` | float | `1e-5` | Privacy parameter used alongside `epsilon` in the Gaussian mechanism. |
| `num_teachers` | int | `10` | Number of noisy discriminator updates run per outer training iteration. |
| `niter` | int | `10000` | Total number of outer training iterations. |
| `batch_size` | int | `128` | Batch size for both discriminator and generator updates. |
| `z_dim` | int, optional | `None` | Latent dimension; defaults to `max(input_dim // 4, 2)` when omitted. |
| `learning_rate` | float | `1e-4` | Adam learning rate for both networks. |
| `lambda_gp` | float | `10.0` | WGAN-GP gradient penalty coefficient. |
| `random_state` | int | `42` | Random seed for reproducibility. |

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
    batch_size=128,
    learning_rate=1e-4,
    random_state=42,
)

model.fit(X_train, y_train, verbose=1)

synthetic_data = model.sample(n_samples=1000)
```

## Katabatic Integration

Like the other models, `train(data_dir, *, synthetic_dir=None, artifact_state_dir=None, **kwargs)`
reads `x_train.csv`/`y_train.csv` from `data_dir`, calls `fit()`, and writes
`x_synth.csv`/`y_synth.csv`/`metadata.json` to `synthetic_dir`. It's registered in
`ModelRegistry` (`supported: True`) and reachable via `ModelRegistry.load_model("pategan")` or
`pip install katabatic[pategan]`.

## Implementation Notes

Where this diverges from the ICLR 2019 paper and its released reference implementation:

- **No literal teacher ensemble.** The paper trains `num_teachers` independent classifiers, each
  on a disjoint data partition. This implementation instead reuses a single shared discriminator,
  updated `num_teachers` times per iteration on independently-sampled (not partitioned) batches —
  the privacy noise is injected into the training labels each update rather than aggregated across
  separate teacher models.
- **Gaussian mechanism, not Laplace.** Privacy noise is added via a Gaussian mechanism
  (`PrivacyMechanism.add_gaussian_noise`, `katabatic/models/pategan/utils.py`), not the paper's
  Laplace-noised vote aggregation.
- **WGAN-GP, not weight clipping.** The discriminator loss uses a gradient penalty term
  (`lambda_gp`), not the released source's RMSProp + weight-clipping setup.
- **No moments accountant / runtime privacy-budget stopping condition.** `epsilon`/`delta`
  configure the noise scale up front; training always runs for `niter` iterations rather than
  stopping early once a tracked `epsilon_hat` exceeds `epsilon`.

Treat `epsilon`/`delta` here as noise-scale knobs consistent with the differential-privacy
literature, not as a certified end-to-end privacy guarantee for this specific implementation.
