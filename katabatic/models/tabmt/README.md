# TabMT for Katabatic

## Model Type
Masked transformer for tabular data generation.

## Model Overview
Trains a BERT-style transformer to reconstruct randomly masked columns of a
row (a variable fraction of columns masked per training batch), then
generates new rows by starting from an all-masked row and iteratively
unmasking columns one at a time in a random order, each prediction
conditioned on the columns already revealed — order-agnostic autoregressive
sampling. The target column is modelled jointly with the features as just
another column.

---

## Research Paper
Gulati, M. and Roysdon, P. (2023)
TabMT: Generating Tabular Data with Masked Transformers
NeurIPS 2023 · https://arxiv.org/abs/2312.06089

---

## Status: simplified implementation

This is a **working but simplified** version of TabMT, not a full
reproduction of the paper. What's simplified, honestly:

- The paper uses a distribution-aware continuous embedding for numeric
  columns; this implementation quantile-bins every numeric column into a
  discrete vocabulary (same tokenization idea used by `bayesian_network`'s
  discretizer) so one shared embedding/output-head mechanism covers both
  column types uniformly. This bounds numeric precision by `n_bins`.
- Sampling reveals columns in one shared random order per `sample()` call, not a
  fully independent random order per row.
- No sample-time confidence-based reordering (the paper's "always reveal the
  most-confident remaining column next" refinement); this always follows a
  fixed random permutation.

The core mechanism — mask a random subset of columns during training,
generate by iteratively unmasking one column at a time at inference — is real
and implemented, and the model trains and samples end-to-end.

---

## Implementation Details
- `utils.ColumnTokenizer`: quantile-bins numeric columns, label-encodes
  categorical columns (target column included), each column gets its own
  vocabulary plus a dedicated MASK id
- `utils.MaskedTabularTransformer`: per-column embedding table, learned
  column-position embedding, shared `nn.TransformerEncoder` backbone,
  per-column linear output head
- Training: uniform-random per-batch mask ratio between `min_mask_ratio` and
  `max_mask_ratio`, cross-entropy loss computed only on masked positions
- Sampling: start fully masked, reveal columns one at a time in a random
  permutation, each step conditioned on all previously-revealed columns

---

## Katabatic Model Structure

katabatic/models/tabmt/

- `__init__.py` → exposes `Tabmt`
- `models.py` → masked-training loop and order-agnostic sampling
- `utils.py` → column tokenizer, masked transformer backbone

---

## Dependencies

`torch` only, already a Katabatic dependency, no new packages required.

```bash
pip install katabatic[tabmt]
```

---

## Hyperparameter Comparison

| Parameter | Class default (`Tabmt._defaults`) | **Benchmark scripts** |
|---|---|---|
| steps | 300 | **3000** |
| batch_size | 128 | **256** |
| d_model | 64 | **128** |
| n_layers | 2 | **3** |
| n_heads | 4 | **4** |
| n_bins | 12 | **12** |
| lr | 1e-3 | **1e-3** |
| mask ratio (min / max) | 0.15 / 0.9 | **0.15 / 0.9** |
| seed | 42 | **42** |
| categorical_cols | inferred from dtype / `info.json` | **passed explicitly from the dataset metadata** |

Passing `categorical_cols` matters: the benchmark runner gives every model integer-coded columns, so inferring roles from dtype would treat categorical codes as numeric. Numeric columns that are integer-valued in training are rounded after decoding (bin midpoints are otherwise fractional).

---

## Usage

```python
from katabatic.models.tabmt.models import Tabmt

model = Tabmt(config={"steps": 3000, "d_model": 128, "n_layers": 3,
                      "categorical_cols": ["workclass", "education"]})  # optional
model.train("path/to/split_dir", synthetic_dir="path/to/synthetic_dir")
synthetic_df = model.sample(len(real_df))
```

`train()` reads `x_train.csv` and `y_train.csv` and writes `x_synth.csv` / `y_synth.csv`.

---

## Model Evaluation Benchmark Results

Evaluated with the Katabatic evaluation pipeline (six dimensions, seed 42). Results below are the runs completed so far; Shuttle is still to be added.

| Dataset | Composite | Fidelity | Utility | Diversity | Privacy | Consistency | Stability |
|---|---|---|---|---|---|---|---|
| Car | 0.8951 | 0.9728 | 0.9894 | 0.9988 | 0.4496 | 0.8998 | 0.9670 |
| Nursery | 0.8863 | 0.9874 | 0.9788 | 0.9990 | 0.4540 | 0.7922 | 0.9925 |
| Magic | 0.8562 | 0.9202 | 0.8949 | 0.6834 | 0.9682 | 0.5016 | 0.9850 |
| Adult | 0.8511 | 0.8815 | 0.8738 | 0.7522 | 0.9669 | 0.5509 | 0.9905 |
| Shuttle | pending | | | | | | |

---

## Model Performance

> **Hardware and runtime:** CPU only. Runtime covers preprocessing, training, sampling and all six evaluation dimensions.

| Dataset | Runtime (s) |
|---|---|
| Car | 102.5 |
| Nursery | 213.2 |
| Magic | 311.3 |
| Adult | 793.1 |
| Shuttle | pending |

Hardware: macOS (Darwin 25.6.0), Apple Silicon (arm64), 17.2 GB RAM, no CUDA GPU.

---

## Strengths
- Handles numeric and categorical columns uniformly through one token vocabulary per column.
- Very high fidelity, utility and diversity on the small categorical datasets (Car, Nursery).
- Trains from scratch with no pretrained weights.

## Limitations
- Simplified relative to the paper; see Status above.
- **Consistency is weak on the larger numeric datasets** (about 0.50 on Magic, 0.55 on Adult).
- **Privacy is low on the small categorical datasets** (about 0.45 on Car and Nursery). The cause was not investigated.
- Numeric precision is bounded by `n_bins` (12 by default).
- Single seed and a fixed step budget; not tuned per dataset.
