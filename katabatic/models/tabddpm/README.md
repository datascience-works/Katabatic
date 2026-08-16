# TabDDPM for Katabatic

## Model Type
Denoising diffusion probabilistic model for synthetic tabular data generation.

## Model Overview
TabDDPM applies Gaussian diffusion to numerical columns and multinomial diffusion to
categorical columns, learning to reverse a noising process over the joint tabular
distribution. Generation works directly on the raw data — there is no pretrained
model hub dependency and no external service call, which keeps the model fully
self-contained inside the Katabatic framework.

Within Katabatic, TabDDPM is used to generate synthetic datasets in the standard
format required for benchmarking and evaluation using the Train Synthetic Test Real
(TSTR) protocol.

---

## Research Paper
Kotelnikov, A., Baranchuk, D., Rubachev, I. and Babenko, A. (2023)
TabDDPM: Modelling Tabular Data with Diffusion Models
Proceedings of ICML 2023
https://arxiv.org/abs/2209.15421

---

## Official GitHub Repository
https://github.com/yandex-research/tab-ddpm

---

## Implementation Details
This implementation lives natively in `katabatic/models/tabddpm/` and follows the
Katabatic 3-file structure. It prefers an external `tabddpm` package if present, and
otherwise falls back to a local, dependency-free `GaussianMultinomialDiffusion` /
`MLPDiffusion` implementation in `utils.py` — so the model runs with only the
dependencies already declared for Katabatic (`torch`), no external model hub or
package required.

Key characteristics:
- Gaussian diffusion over numerical columns, multinomial diffusion over categorical
  columns, combined in a single denoising objective
- MLP-based denoiser (`d_layers` configurable), cosine or linear noise scheduler
- `train()` supports two modes: array mode (`X`, `y`) and pipeline mode
  (`dataset_dir` + `synthetic_dir`, matching the Katabatic pipeline contract)
- `sample()` reconstructs a DataFrame with original column order and decodes
  categorical columns back to their original labels

---

## Katabatic Model Structure

katabatic/models/tabddpm/

- `__init__.py` → exposes `Tabddpm`
- `models.py` → main model, training loop, sampling logic
- `utils.py` → local diffusion implementation (Gaussian + multinomial), fallback for
  the external `tabddpm` package

---

## Dependencies

No dependencies outside the Katabatic framework's own extras. Install with:

```bash
pip install katabatic[tabddpm]
```

which resolves to `torch`, `scipy`, `rtdl_revisiting_models`, `category-encoders` as
declared in `pyproject.toml`.

---

## Dataset Format (Input)

sample_data/<dataset_name>/

- x_train.csv
- y_train.csv
- x_test.csv
- y_test.csv

---

## Output Format (Generated)

synthetic/<dataset_name>/tabddpm/

- x_synth.csv
- y_synth.csv

---

## Datasets Used

- CAR
- MAGIC
- NURSERY
- ADULT
- SHUTTLE

---

## Running the Model (Example)

```python
from katabatic.models.tabddpm.models import Tabddpm

model = Tabddpm()

model.train(
    "sample_data/car",
    synthetic_dir="synthetic/car/tabddpm",
)
```

Or via the batch runner covering all five acceptance-criteria datasets:

```bash
bash scripts/run_tabddpm.sh
```

---

## Pipeline Steps

1. Load `x_train.csv` / `y_train.csv` from `sample_data/<dataset>/`
2. Train the Gaussian + multinomial diffusion model
3. Sample synthetic rows and decode back to original column types
4. Write `x_synth.csv` / `y_synth.csv` to `synthetic/<dataset>/tabddpm/`
5. Run TSTR evaluation against `sample_data/<dataset>/x_test.csv` / `y_test.csv`

---

## Status

**In progress.** Core model code already existed in `main`/registry but had no
README, no per-dataset run scripts, and no evaluation runs — this submission adds
all three plus a first end-to-end smoke pass. Full benchmark-quality runs (production
`steps`/`num_timesteps` config, not the lightweight smoke config) are the next step,
tracked in `scripts/run_tabddpm.sh`.

## Important Notes

- The lightweight config in `scripts/run_tabddpm.sh` (`steps`, `num_timesteps`
  reduced) is for a fast correctness smoke test, not a quality benchmark — full runs
  need the defaults in `Tabddpm._defaults`.
- Categorical columns are label-encoded internally; unseen indices at sample time are
  clipped for safety.
