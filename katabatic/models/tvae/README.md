# TVAE

A variational autoencoder for synthetic tabular data generation, adapted for the Katabatic framework.

## Paper

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
**"Modeling Tabular data using Conditional GAN."**
*Advances in Neural Information Processing Systems (NeurIPS).*

- Paper: https://arxiv.org/abs/1907.00503
- Official code: https://github.com/sdv-dev/CTGAN

## Why this model

Most tabular generative models in Katabatic (CTGAN, MedGAN, FairTabDiffusion) rely on adversarial or diffusion-based training, both of which can be unstable or slow to converge. TVAE takes a different approach:

1. **Single-objective training** — TVAE optimizes a reconstruction loss plus a KL-divergence regularizer, avoiding the adversarial dynamics (mode collapse, discriminator/generator imbalance) that GAN-based models are prone to.
2. **Latent-space generation** — synthetic rows are decoded from samples drawn from the learned latent distribution, giving a smooth, continuous generative space rather than a step-wise denoising process.

This makes TVAE a useful comparison point against FairTabDiffusion's diffusion-based approach, particularly for investigating whether the continuous-feature consistency issues found in FairTabDiffusion (Adult, Magic, Shuttle) are diffusion-specific or common across generative architectures.

## Implementation notes

This Katabatic port follows the same `Model` contract and on-disk conventions as `CTGANModel` (`katabatic/models/ctgan/models.py`) and `FairTabDiffusion` (`katabatic/models/fairtabdiffusion/models.py`):

- `train(data_dir, synthetic_dir=None)` reads `train_full.csv` (or `x_train.csv` + `y_train.csv`) from `data_dir`, trains the model, then writes `x_synth.csv`, `y_synth.csv`, and `metadata.json` to `synthetic_dir`.
- `evaluate()` is a placeholder returning `0.0` (actual evaluation is handled by Katabatic's evaluation pipeline).
- `sample(n, conditional=None)` generates `n` synthetic rows. Unlike `FairTabDiffusion`, conditional sampling is **not** supported by the underlying `ctgan.TVAE` synthesizer — passing a non-`None` `conditional` argument raises a `ValueError`.

The underlying synthesizer is the reference `TVAE` implementation from the `ctgan` package (sdv-dev), which shares CTGAN's data transformer for handling mixed categorical/continuous columns. Discrete columns are auto-detected (object dtype or fewer than 20 unique values) if not otherwise specified, matching the discretized/integer-encoded convention used elsewhere in Katabatic.

## Usage

```python
from katabatic.models.tvae.models import TVAE

model = TVAE(
    epochs=300,
    batch_size=256,
    embedding_dim=128,
    seed=42,
)

model.train(data_dir="sample_data/adult", synthetic_dir="synthetic/adult/tvae")

# Optional: generate additional samples after training
synthetic_df = model.sample(n=1000)
```

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `300` | Training epochs. |
| `batch_size` | `500` | Minibatch size. |
| `embedding_dim` | `128` | Size of the latent embedding space. |
| `compress_dims` | `(128, 128)` | Encoder network layer sizes. |
| `decompress_dims` | `(128, 128)` | Decoder network layer sizes. |
| `seed` | `42` | Random seed. |

## Datasets tested

Adult, Car, Magic, Nursery, Shuttle (standard Katabatic benchmark set) — same five datasets used for FairTabDiffusion, to allow direct composite-score and per-dimension comparison across the two model architectures.

## Dependencies

- `ctgan` (new optional dependency; not currently used by any other model in the repository — register as `tvae = ["ctgan>=0.10.0"]` in `pyproject.toml`).

## Computational complexity

Low. Single-objective VAE training (no adversarial or multi-step diffusion process); typically faster to converge than CTGAN or FairTabDiffusion on the standard benchmark datasets.

## Status

Initial implementation — pending PEP 8/Ruff formatting pass, cross-validation, and benchmark runs across all five standard datasets before PR submission to `development`.
