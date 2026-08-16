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
- Sampling reveals columns in one shared random order per batch, not a fully
  independent random order per row.
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

`torch` only — already a Katabatic dependency, no new packages required.

```bash
pip install katabatic[tabmt]
```

---

## Dataset Format (Input) / Output Format (Generated)

Same as the other models: `sample_data/<dataset>/{x,y}_{train,test}.csv` in,
`synthetic/<dataset>/tabmt/{x,y}_synth.csv` out.

## Datasets Used

- CAR, MAGIC, NURSERY, ADULT, SHUTTLE

## Running the Model (Example)

```python
from katabatic.models.tabmt.models import Tabmt

model = Tabmt(config=dict(steps=300))
model.train("sample_data/car", synthetic_dir="synthetic/car/tabmt")
```

Or via the batch runner:

```bash
MODEL=tabmt bash scripts/run_new_models.sh
```

## Status: run history

Ran end-to-end (train → sample → TSTR evaluate) on all 5 datasets using a
lightweight smoke config (`steps=150`). Results in
`Results/<dataset>/tabmt_tstr.csv`. Masked-transformer training needs more
steps than the GAN/diffusion models here to converge (mask reconstruction is
a harder objective at low step counts) — that's the next step, not a
correctness bug.
