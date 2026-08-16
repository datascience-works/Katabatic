# CTAB-GAN+ for Katabatic

## Model Type
Conditional Generative Adversarial Network (GAN) for synthetic tabular data generation.

## Model Overview
CTAB-GAN+ is a conditional GAN designed for skewed, long-tailed and imbalanced
tabular columns. It combines mixed-type encoding (mode-specific normalisation for
continuous columns, one-hot/log-transform handling for categorical and long-tailed
columns) with a training-by-sampling conditional vector scheme, and trains with
WGAN-GP for stability.

---

## Research Paper
Zhao, Z., Kunar, A., Birke, R. and Chen, L. Y. (2022)
CTAB-GAN+: Enhancing Tabular Data Synthesis
https://arxiv.org/abs/2204.00401

## Official GitHub Repository
https://github.com/Team-TUD/CTAB-GAN-Plus

---

## Status: In Progress

This model does **not** have a working implementation yet. What exists as of this
branch:

- Directory scaffold in the required 3-file Katabatic structure
  (`__init__.py`, `models.py`, `utils.py`)
- `CtabganPlus` class stub implementing the correct `Model` interface
  (`train`/`evaluate`/`sample`, pipeline-mode `dataset_dir`/`synthetic_dir`
  signature) so it plugs into the existing pipeline/evaluation code without
  interface changes once the training loop is written
- A dependency-compliance audit (see below) that changed the implementation plan

### Audit finding: prior unmerged attempt is mislabeled and non-compliant

An earlier attempt exists on the unmerged remote branch `feature/ctabgan_plus`
(commit `5f33883`, "Add CTABGAN+ model"). Its README documents a `CTABGANPlus`
class, but the actual `models.py` on that branch defines a `CopulaGANModel` class
that wraps `sdv.single_table.CopulaGANSynthesizer` — SDV's CopulaGAN, not the
CTAB-GAN+ architecture from the paper — and imports `sdv`, a package outside the
Katabatic framework's own dependency set.

This fails two of the model-acceptance criteria directly:
- **"The model does not have external dependencies outside the Katabatic
  framework"** — `sdv` is not a Katabatic dependency or extra.
- The README/code mismatch (`CTABGANPlus` documented vs. `CopulaGANModel`
  implemented) means it also would not pass a documentation-accuracy check.

Given that, this submission does not merge or build on that branch. Implementation
is being written from scratch against the CTAB-GAN+ paper directly.

---

## Implementation Plan

- [ ] Mixed-type encoder (VGM for continuous, one-hot/log-transform for
      categorical & long-tailed columns) — in `utils.py`
- [ ] Conditional vector sampler with training-by-sampling
- [ ] Generator / discriminator trained with WGAN-GP
- [ ] Wire `train()`/`sample()` to the pipeline-mode contract already stubbed
      in `models.py`
- [ ] `scripts/run_ctabgan_plus.sh` across the 5 acceptance-criteria datasets
- [ ] TSTR evaluation pass, `Results/<dataset>/ctabgan_plus_tstr.csv`

---

## Katabatic Model Structure

katabatic/models/ctabgan_plus/

- `__init__.py` → exposes model
- `models.py` → main model class (interface complete, training loop pending)
- `README.md` → this file

---

## Dependencies

`torch` only — already a Katabatic dependency, no new packages required. This is a
deliberate constraint versus the prior attempt, which pulled in `sdv`.

---

## Dataset Format (Input)

sample_data/<dataset_name>/

- x_train.csv
- y_train.csv
- x_test.csv
- y_test.csv

## Output Format (Generated, once implemented)

synthetic/<dataset_name>/ctabgan_plus/

- x_synth.csv
- y_synth.csv

## Datasets Used (target)

- CAR
- MAGIC
- NURSERY
- ADULT
- SHUTTLE
