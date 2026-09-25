# TVAE-GAN

## Model Overview

TVAE-GAN generates synthetic tabular data by combining a Variational Autoencoder with a Generative Adversarial Network. A single shared network acts as both the VAE's decoder and the GAN's generator, and reconstruction quality is measured using the discriminator's internal
features rather than raw data values, a "learned similarity metric" rather than a fixed one like MSE.

This implementation was built from scratch to match the original paper's architecture, since no working implementation existed in this project's main codebase.

### Key Idea

Instead of training a VAE with a simple reconstruction loss (like pixel-wise MSE), TVAE-GAN measures reconstruction quality using a discriminator's learned feature representations. The same network that decodes latent vectors back into data is also used as the GAN's generator, one network, two roles. This means the model learns what "similar" actually looks like for the data, rather than assuming a fixed distance metric captures it.

### Research Paper

**Autoencoding beyond pixels using a learned similarity metric**

Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther
ICML 2016
Paper: https://arxiv.org/abs/1512.09300

**What was kept the same, what was changed:**

This implementation follows the paper's core architecture, a VAE encoder, a single shared decoder/generator network, and a discriminator whose hidden-layer features are used for the reconstruction loss, along with the paper's three-part training objective (KL prior discriminator-feature reconstruction, adversarial loss).
The main change is the data type: the paper is written for 64x64 images (CelebA faces) using convolutional networks, this implementation uses simple MLPs, since tabular columns have no spatial structure to convolve over.

**Parts not specified in the paper:**

The paper doesn't give a specific numeric value for gamma (the weighting between reconstruction and adversarial loss in the decoder's update, Eq. 9), it's described but left as a tunable choice. This implementation defaults gamma to 1.0, an inferred value, not something stated in the paper. Similarly, the paper doesn't specify which discriminator layer to use for feature-based reconstruction loss when the network isn't convolutional, this implementation uses the middle hidden layer as a reasonable default.

## Approach
TVAE-GAN trains three networks together:
1. **Encoder**: maps a real row to latent distribution parameters, then samples a latent vector via the reparameterization trick.

2. **Decoder/Generator**: one shared network reconstructs real rows from their latent code, and generates new rows from random latent vectors.

3. **Discriminator**: tells real rows apart from reconstructed or generated ones, and its internal features are used to judge
   reconstruction quality.

### Training Details

All three networks train each batch, but each learns only from the loss terms that apply to it, the Encoder from KL prior + reconstruction loss, the Decoder/Generator from a weighted mix of reconstruction and fooling the discriminator, and the Discriminator only from telling real, reconstructed, and generated rows apart. Keeping these separate matters, otherwise the discriminator would collapse toward treating everything as the same.

### Convergence Criteria

Runs for a fixed number of epochs (default 50). No early stopping yet, training length was found to affect result quality during testing (see Known Issues).

## Hyperparameters
Found in `katabatic/models/tvaegan/models.py` (`TVAEGANModel.__init__`).

| Parameter | Default | Notes |
|---|---|---|
| `latent_dim` | 32 | Not specified in the paper for tabular data, chosen as a reasonable size |
| `hidden_dims` | [128, 64] | Not specified in the paper for tabular data |
| `epochs` | 50 | Not tuned yet |
| `batch_size` | 64 | Matches paper (Section 4) |
| `lr` | 3e-4 | Matches paper's 0.0003 (Section 4) |
| `gamma` | 1.0 | Paper describes this weighting (Eq. 9) but doesn't give a specific value, this is an inferred default |
| `seed` | 42 | For reproducibility |
| `discriminator_hidden_dims` | None (uses hidden_dims) | Testing showed a smaller discriminator ([32,16] vs default) significantly improves consistency and resolves class collapse on imbalanced datasets |

## Input
- `X`: Tabular feature matrix (numeric and/or categorical columns)
- `y`: Target labels

### Expected Files
- `x_train.csv`, `y_train.csv` in the data directory

## Preprocessing
### Numerical Features
Numeric columns are standardised (mean 0, standard deviation 1).

### Categorical Features
Categorical columns (including the target, forced as categorical) are one-hot encoded.

## Label Handling
The target column is treated as a forced categorical column and combined with the feature columns before encoding, it isn't handled specially elsewhere in the pipeline.

## Output
Generated files (when using `train()`):
- `x_synth.csv`
- `y_synth.csv`
- `synthetic.csv` (combined)

## Evaluation
`evaluate()` returns a reconstruction loss, measured in the discriminator's feature space rather than raw data space (matching the
paper's own similarity metric, Eq. 6-7). Lower is better.

The full evaluation used for validation is `SyntheticEvaluationPipeline`, which reports 6 dimensions (fidelity, utility, diversity, privacy, consistency, stability) combined into a composite score.

## Strengths
- Learns a data-driven similarity metric instead of assuming a fixed one, which can capture relationships a simple MSE would miss
- Ran cleanly through all 6 evaluation dimensions in testing, including stability, which some other models in this project haven't managed
- Very stable results across different random seeds (stability score 0.97 in testing)

## Limitations
- Consistency remains poor (discriminator detects synthetic data ~99.96% of the time on Adult), though reducing the discriminator's capacity relative to the encoder/decoder (discriminator_hidden_dims=[32,16]) substantially improved overall results: composite score 0.43 to 0.79, utility went from crashing to 0.89, consistency nearly doubled. Not fully resolved.
- Severe mode collapse can occur on imbalanced datasets under default
  settings ,resolved via more training epochs and/or a smaller discriminator.
- No early stopping implemented yet

## Installation
No extra dependencies required beyond the core project setup
(uses `torch` and `sklearn`, already required elsewhere).

## Usage
Benchmark scripts for each dataset:
- Car: [benchmarks/examples/tvaegan/run_tvaegan_car.py](../../../../benchmarks/examples/tvaegan)

```python
from katabatic.models.tvaegan.models import TVAEGANModel

model = TVAEGANModel()

model.train(
    dataset_dir="path_to_data",
    synthetic_dir="path_to_save"
)

synthetic_df = model.sample(1000)
```

## Model Evaluation Benchmarks Results

#### Car Dataset (default configuration)

Composite score: 0.7149

Dimension scores:
- fidelity       0.8036
- utility        0.7918
- diversity      0.9501
- privacy        0.4751
- consistency    0.2201, flagged as a known issue
- stability      0.9725

## Model Performance Benchmarks Results
Training and inference timing not yet measured for this model.
