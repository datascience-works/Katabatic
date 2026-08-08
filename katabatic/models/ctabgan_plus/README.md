# CTAB-GAN+ Model

## Model Overview
CTAB-GAN+ is a conditional GAN for synthetic tabular data generation, implemented from
scratch for the Katabatic framework using PyTorch. It follows the architecture described in
**"CTAB-GAN+: Enhancing Tabular Data Synthesis" by Zhao et al. (2022)**
([arXiv:2204.00401](https://arxiv.org/abs/2204.00401)), using a CNN-based generator and
discriminator, an auxiliary classifier for downstream task consistency, and column-wise
data transforms (`DataPrep`, `DataTransformer`, `ImageTransformer`) tailored to mixed
tabular data types.

---

## Dependencies
Base Katabatic dependencies (`pandas`, `numpy`, `scikit-learn`) plus:
```
pip install torch
```

---

## Usage

```python
from katabatic.models.ctabgan_plus.models import CTABGANPlus

model = CTABGANPlus(config={"epochs": 200})

model.train(
    dataset_dir="sample_data/magic",
    synthetic_dir="synthetic/magic/ctabgan_plus",
    categorical=[10]  # column indices, aligned to x_train + y_train combined; label column must be included
)

X_synth = model.sample(1000)
```

**Constructor:** takes a single `config` dict (all keys optional):

| Key | Default | Description |
|---|---|---|
| `class_dim` | `(256, 256, 256, 256)` | Auxiliary classifier layer sizes |
| `random_dim` | `100` | Generator noise vector size |
| `num_channels` | `64` | Base CNN channel count |
| `l2scale` | `1e-5` | Weight decay for both optimizers |
| `batch_size` | `500` | Training batch size |
| `epochs` | `300` | Training epochs |

Optional column-typing hints can also be passed via `config`: `log_columns`, `mixed_columns`,
`general_columns`, `non_categorical_columns`, `integer_columns` — see `DataPrep` in `utils.py`
for details.

**`train()` requires `categorical` to be provided explicitly** — it raises `ValueError` if
omitted, and `TypeError`/`ValueError` if the indices aren't valid integer column positions
within the combined `x_train` + `y_train` dataframe (label column included). This is by
design, not a bug.

**`evaluate()` deliberately raises `NotImplementedError`** — evaluation for this model is
handled externally by Katabatic's TSTR benchmarking pipeline, not by the model itself.

---

## ⚠️ Note: a second CTAB-GAN+ implementation exists elsewhere in the codebase
The `Katabatic-feature-ctabgan_plus` branch contains a different, SDV-wrapped
`CopulaGANModel` under the same `ctabgan_plus` module path. That version's constructor,
`train()` signature, and `evaluate()` behavior are **not compatible** with this
implementation — don't mix usage examples between the two. This implementation (full
custom PyTorch, matching the original paper) appears to be the intended long-term version;
the team should confirm which one is canonical and consider deprecating the other.

---

## Validated Behaviour (Trimester 2 2026 validation pass)
Tested end-to-end on the `magic` dataset (`epochs=200`, `categorical=[10]`):

- ✅ Trains and generates synthetic data successfully, no errors
- ✅ Label distribution closely matches real data (synthetic ~67/33 vs real ~65/35 split)
- ⚠️ Synthetic feature-level variance is consistently lower than real data across all 10
  features (standard deviation reduced by roughly 35–55%). Value ranges and central
  tendency are preserved, but the model under-represents the true spread/diversity of the
  real data — may benefit from more training epochs or architecture/hyperparameter tuning
- ⚠️ No training progress is visible in the console. The code calls `logger.info(...)` per
  epoch, but `logging.basicConfig()` is never called anywhere in the model, so these
  messages are silently dropped at the default logging level. Callers who want visibility
  should call `logging.basicConfig(level=logging.INFO)` before training
- Training is noticeably slower than lighter-weight tabular GAN approaches, due to the
  CNN-based architecture and CPU-only execution (no GPU available in this environment)
