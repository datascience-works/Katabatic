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
- `evaluate(real_df, target_col=..., test_data=...)`, inherited from `Model`, scores the fitted model on Katabatic's six dimensions (fidelity, utility via TSTR, diversity, privacy, consistency and stability) and returns an `EvaluationReport` (`report.dimension_scores`, `report.composite_score`).

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

## Benchmark results

| Dataset | Fidelity | Utility | Diversity | Privacy | Consistency | Stability | Composite | Runtime |
|---|---|---|---|---|---|---|---|---|
| Car | 0.8664 | 0.8473 | 0.9752 | 0.4597 | 0.8589 | 0.9590 | **0.8135** | 41 s |
| Nursery | 0.9006 | 0.8622 | 0.9806 | 0.4683 | 0.8575 | 0.9830 | **0.8301** | 3 min |
| Magic | 0.9403 | 0.9362 | 0.9060 | 0.7090 | 0.5784 | 0.9875 | **0.8669** | 7 min |
| Shuttle | 0.9345 | 0.9421 | 0.9710 | 0.8747 | 0.3848 | 0.9805 | **0.8792** | 9 min |
| Adult | 0.9473 | 0.8866 | 0.9955 | 0.8588 | 0.5032 | 0.9940 | **0.8755** | 14 min |

How to read these scores:

| Dataset | Exact duplicates of real rows | Near-duplicates (Gower < 0.01) | Real-vs-synthetic classifier accuracy | TSTR / TRTR accuracy |
|---|---|---|---|---|
| Car | 81% | 0% | 0.58 | 0.683 / 0.861 |
| Nursery | 80% | 0% | 0.54 | 0.745 / 0.896 |
| Magic | 0% | 72% | 0.75 | 0.754 / 0.827 |
| Shuttle | 0% | 32% | 0.92 | 0.906 / 0.978 |
| Adult | 0% | 23% | 0.83 | 0.696 / 0.836 |

- **The exact duplicates on car and nursery aren't memorisation.** Both datasets list every possible feature combination exactly once, so any valid synthetic row matches a real one. With an 80/20 split, about 80% of rows would match the training data by chance, which is what the model produces.
- **Balanced sampling can lower TSTR on imbalanced datasets.** Labels are drawn uniformly by default, so the synthetic class mix differs from the real test set. Use `balanced_sampling=False` to keep the real class distribution.
- **Consistency is the weakest dimension** on magic, shuttle and adult (0.38–0.58).
- GPU memory peaked at under 0.5 GB on every dataset.

## Dependencies

- `torch`, installed via the `fairtabdiffusion` extra:

```bash
pip install katabatic[fairtabdiffusion]   # or: poetry install -E fairtabdiffusion
```

## Computational complexity

Low–Medium. A lightweight MLP-based DDPM (not U-Net based); trains comparably fast to TabDDPM on the standard benchmark datasets.

## Status

Officially supported (`supported: True` in `ModelRegistry`), with an artifact-pipeline integration test in `tests/test_integration_fairtabdiffusion.py`.
