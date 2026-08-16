# TabDiff for Katabatic

## Model Type
Denoising diffusion model with a unified (joint) diffusion process over
numeric and categorical columns.

## Model Overview
Where TabDDPM (already in this repo) runs two separate diffusion processes —
Gaussian for numeric columns, multinomial for categorical columns — TabDiff's
distinguishing idea is to diffuse the *entire* joint feature space with a
single process. This implementation follows that idea directly: categorical
columns are one-hot encoded, numeric columns z-scored, everything concatenated
into one continuous vector, and a single class-conditional Gaussian DDPM
diffuses/denoises that joint vector. Categorical blocks are recovered with
argmax at sampling time.

---

## Research Paper
Shi, J., Xu, M., Hua, H., Zhang, H., Ermon, S. and Leskovec, J. (2024)
TabDiff: A Multi-Modal Diffusion Model for Tabular Data Generation
ICLR 2025 · https://arxiv.org/abs/2410.20626

---

## Status: simplified implementation

This is a **working but simplified** version of TabDiff, not a full
reproduction of the paper. What's simplified, honestly:

- The paper learns per-feature-type **adaptive noise schedules** (different
  diffusion dynamics for numeric vs. categorical dimensions within the joint
  process); this implementation uses one shared cosine schedule across the
  whole joint vector.
- The paper's denoiser is a transformer over per-column tokens; this uses a
  small MLP over the flattened joint vector — sufficient to demonstrate the
  unified-diffusion mechanism, but with less per-column inductive bias.
- Categorical columns are recovered with hard argmax rather than the paper's
  learned discretization step.

The core mechanism that distinguishes TabDiff from TabDDPM — one diffusion
process over the joint space rather than two separate ones — is real and
implemented, and the model does train (loss decreases) and sample end-to-end.

---

## Implementation Details
- `utils.JointEncoder`: z-scores numeric columns, one-hot encodes categorical
  columns, concatenates into one continuous vector; `info.json`-aware column
  role detection (same approach as the other new models in this branch)
- `utils.DenoiserMLP`: small MLP conditioned on diffusion timestep and class
  label, no other dependency on the target column's semantics
- `utils.cosine_beta_schedule`: standard cosine noise schedule
- `train()` / `sample()` follow the same `dataset_dir` / `synthetic_dir`
  pipeline-mode contract as `tabddpm` and `ctgan`

---

## Katabatic Model Structure

katabatic/models/tabdiff/

- `__init__.py` → exposes `Tabdiff`
- `models.py` → joint diffusion training loop and reverse sampling
- `utils.py` → joint encoder, denoiser MLP, noise schedule

---

## Dependencies

`torch` only — already a Katabatic dependency, no new packages required.

```bash
pip install katabatic[tabdiff]
```

---

## Dataset Format (Input) / Output Format (Generated)

Same as the other models: `sample_data/<dataset>/{x,y}_{train,test}.csv` in,
`synthetic/<dataset>/tabdiff/{x,y}_synth.csv` out.

## Datasets Used

- CAR, MAGIC, NURSERY, ADULT, SHUTTLE

## Running the Model (Example)

```python
from katabatic.models.tabdiff.models import Tabdiff

model = Tabdiff(config=dict(steps=300, num_timesteps=100))
model.train("sample_data/car", synthetic_dir="synthetic/car/tabdiff")
```

Or via the batch runner:

```bash
MODEL=tabdiff bash scripts/run_new_models.sh
```

## Status: run history

Ran end-to-end (train → sample → TSTR evaluate) on all 5 datasets using a
lightweight smoke config (`steps=150`, `num_timesteps=60`) — enough to prove
the pipeline is correct, not enough for benchmark-quality synthetic data.
Results in `Results/<dataset>/tabdiff_tstr.csv`. TSTR accuracy at this smoke
config is noticeably lower than the other models in this branch; increase
`steps`/`num_timesteps` toward TabDDPM's defaults for a real quality pass —
that's the next step here, not a correctness bug.
