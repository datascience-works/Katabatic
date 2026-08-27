# FairTabDiffusion

A fairness-aware conditional diffusion model for synthetic tabular data generation, adapted for the Katabatic framework.

## Paper

Yang, Z., Yu, H., Guo, P., Zanna, K., Yang, X., & Sano, A. (2024).
**"Balanced Mixed-Type Tabular Data Synthesis with Diffusion Models."**
*Transactions on Machine Learning Research (TMLR).*

- Paper: https://arxiv.org/abs/2404.08254
- Official code: https://github.com/comp-well-org/fair-tab-diffusion

## Why this model

Standard tabular diffusion models (e.g. TabDDPM, already in Katabatic) can inherit and amplify demographic imbalance present in the training data — for example, generating synthetic records that under-represent a minority group relative to the real dataset. FairTabDiffusion addresses this directly by:

1. **Conditioning** the denoising network on both the target label and a chosen **sensitive attribute** (e.g. sex, race) during training.
2. **Balanced sampling** at generation time — the label and sensitive attribute are drawn *uniformly* rather than from their empirical (possibly skewed) distribution, so the synthetic dataset has a fair joint distribution over `(label, sensitive_attribute)` by construction.

## Implementation notes

This Katabatic port follows the same `Model` contract and on-disk conventions as `CTGANModel` (`katabatic/models/ctgan/models.py`):

- `train(data_dir, synthetic_dir=None)` reads `train_full.csv` (or `x_train.csv` + `y_train.csv`) from `data_dir`, trains the model, then writes `x_synth.csv`, `y_synth.csv`, and `metadata.json` to `synthetic_dir`.
- `evaluate()` is a placeholder returning `0.0` (actual evaluation is handled by Katabatic's evaluation pipeline).
- `sample(n, conditional=None)` generates `n` synthetic rows, optionally conditioned on a fixed label value via `conditional={<label_col>: value}`.

The diffusion process is a lightweight Gaussian DDPM (MLP-based denoiser, not U-Net) operating over a min-max normalized encoding of Katabatic's standard discretized/integer-encoded columns, decoded back to categorical bins by rounding at sampling time. This is a simplification of the original paper's separate Gaussian/multinomial diffusion, chosen to keep the implementation dependency-light (PyTorch only, already used by `codi`, `ctgan`, `medgan`, `tabddpm`, `great`) while preserving the core fairness mechanism: **conditioning on a sensitive attribute + balanced (uniform) sampling at generation time**.

## Usage

```python
from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

model = FairTabDiffusion(
    sensitive_col="sex",   # set to None for datasets with no natural sensitive attribute
    epochs=200,
    timesteps=100,
    batch_size=256,
)

model.train(data_dir="sample_data/adult", synthetic_dir="synthetic/adult/fairtabdiffusion")

# Optional: generate additional samples after training
synthetic_df = model.sample(n=1000)
```

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `sensitive_col` | `None` | Feature column name for the sensitive attribute (e.g. `"sex"`). Falls back to unconditional (label-only) diffusion if not provided or not present in the data. |
| `epochs` | `200` | Training epochs. |
| `batch_size` | `256` | Minibatch size. |
| `timesteps` | `100` | Number of diffusion steps (T). |
| `hidden` | `256` | Hidden width of the denoising MLP. |
| `lr` | `1e-3` | Adam learning rate. |
| `seed` | `42` | Random seed. |
| `device` | `None` (auto) | `"cuda"` or `"cpu"`. |
| `balanced_sampling` | `True` | If `True`, `sample()` draws label/sensitive attribute uniformly (fair generation). If `False`, samples from the empirical training distribution instead. |

## Datasets tested

Adult, Car, Magic, Nursery, Shuttle (standard Katabatic benchmark set). For datasets without an obvious sensitive attribute (Car, Magic, Nursery, Shuttle), leave `sensitive_col=None` — the model degrades gracefully to unconditional (TabDDPM-style) diffusion. For Adult, `sex` or `race` are natural choices for `sensitive_col`.

## Dependencies

- `torch` (already an optional dependency in `pyproject.toml`, used by `codi`, `ctgan`, `medgan`, `tabddpm`, `great`).

## Computational complexity

Low–Medium. A lightweight MLP-based DDPM (not U-Net based); trains comparably fast to TabDDPM on the standard benchmark datasets.

## Status

Initial implementation — pending PEP 8/Ruff formatting pass, cross-validation, and benchmark runs across all five standard datasets before PR submission to `development`.
