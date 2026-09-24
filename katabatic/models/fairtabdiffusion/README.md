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

This Katabatic port follows the standard `Model` contract (`katabatic/models/base_model.py`):

- `train(data_dir, *, categorical_cols=None, continuous_cols=None, synthetic_dir=None, artifact_state_dir=None)` reads `train_full.csv` (or `x_train.csv` + `y_train.csv`) from `data_dir`, trains the model, writes `x_synth.csv`, `y_synth.csv`, and `metadata.json` to `synthetic_dir` when given, and persists fitted state to `artifact_state_dir` for `load_from_ref()`. If `categorical_cols` isn't given, feature types are auto-detected (dtype + low-cardinality integers).
- `sample(n_samples=None, *, conditional=None, seed=None)` generates `n_samples` rows (default: the training row count), features plus the label as the last column. `conditional={<label_col>: value}` fixes the label; `seed` makes the call reproducible.
- `evaluate()` is not implemented (it raises `NotImplementedError`); use Katabatic's evaluation pipeline for metrics.

The diffusion process is a lightweight Gaussian DDPM (MLP-based denoiser, not U-Net). Continuous columns are quantile-transformed to a normal distribution; categorical columns (including the label) are one-hot encoded and decoded by argmax. The denoiser is conditioned on the label and the sensitive attribute through learned embeddings. This is a simplification of the original paper's separate Gaussian/multinomial diffusion, chosen to keep the implementation dependency-light (PyTorch only) while preserving the core fairness mechanism: **conditioning on a sensitive attribute + balanced (uniform) sampling at generation time**.

## Usage

```python
from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

model = FairTabDiffusion(
    sensitive_col="sex",   # set to None for datasets with no natural sensitive attribute
    epochs=200,
    timesteps=100,
    batch_size=256,
)

model.train("sample_data/adult", synthetic_dir="synthetic/adult/fairtabdiffusion")

# Optional: generate additional samples after training
synthetic_df = model.sample(1000, seed=42)
```

Through the artifact pipeline (also persists state for `FairTabDiffusion.load_from_ref`):

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

TrainTestSplitPipeline(model=FairTabDiffusion(sensitive_col="sex")).run(
    input_csv="data.csv",
    dataset_name="mydata",
    artifact_store=store,
    model_name="fairtabdiffusion",
)
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
| `balanced_sampling` | `True` | If `True`, `sample()` draws the label and sensitive attribute uniformly (fair generation). If `False`, both follow their empirical training distribution. |

## Datasets tested

Adult, Car, Magic, Nursery, Shuttle (standard Katabatic benchmark set). For datasets without an obvious sensitive attribute (Car, Magic, Nursery, Shuttle), leave `sensitive_col=None` — the model then conditions on the label only. Note that with the default `balanced_sampling=True` the **label is still rebalanced to uniform**, so the synthetic class distribution will differ from the real one on imbalanced datasets (e.g. Shuttle, Car). Pass `balanced_sampling=False` to keep the real class distribution. For Adult, `sex` or `race` are natural choices for `sensitive_col`.

## Dependencies

- `torch`, installed via the `fairtabdiffusion` extra:

```bash
pip install katabatic[fairtabdiffusion]   # or: poetry install -E fairtabdiffusion
```

## Computational complexity

Low–Medium. A lightweight MLP-based DDPM (not U-Net based); trains comparably fast to TabDDPM on the standard benchmark datasets.

## Status

Officially supported (`supported: True` in `ModelRegistry`), with an artifact-pipeline integration test in `tests/test_integration_fairtabdiffusion.py`.
