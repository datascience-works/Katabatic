# GMM (Gaussian Mixture Model) for Katabatic

## Model Type
Class-conditional density-estimation baseline for synthetic tabular data generation.

## Model Overview
Fits one `sklearn.mixture.GaussianMixture` per target class over a label-encoded
feature matrix (categorical columns label-encoded, numeric columns used as-is),
then generates synthetic rows by sampling from those fitted densities in
proportion to the empirical class distribution. No training loop, no GPU —
this is a classical density-estimation baseline, useful as a fast reference
point against the deep generative models in the registry.

---

## Research Paper
This is a standard statistical baseline (Gaussian Mixture Models), not tied to
a single tabular-synthesis paper. Reference implementation: scikit-learn's
`GaussianMixture` — https://scikit-learn.org/stable/modules/mixture.html

---

## Implementation Details
- Categorical columns are label-encoded to integers, sampled in continuous
  space, then rounded/clipped back to a valid category index on decode
  (`utils.TabularEncoder`)
- Column roles (numeric vs. categorical) are read from each dataset's
  `info.json` where available, falling back to dtype detection otherwise —
  this matters because some datasets (e.g. `car`) encode categorical columns
  as integers, which dtype alone can't distinguish from numeric columns
- One `GaussianMixture` fitted per class; `n_components` capped by the
  smallest class's row count

---

## Katabatic Model Structure

katabatic/models/gmm/

- `__init__.py` → exposes `Gmm`
- `models.py` → main model class
- `utils.py` → column-role loading + numeric/categorical encode-decode

---

## Dependencies

None beyond scikit-learn, which is already a Katabatic core dependency. No
`pip install katabatic[gmm]` extra is required beyond the base install.

---

## Dataset Format (Input)

sample_data/<dataset_name>/

- x_train.csv
- y_train.csv
- x_test.csv
- y_test.csv

## Output Format (Generated)

synthetic/<dataset_name>/gmm/

- x_synth.csv
- y_synth.csv

## Datasets Used

- CAR
- MAGIC
- NURSERY
- ADULT
- SHUTTLE

## Running the Model (Example)

```python
from katabatic.models.gmm.models import Gmm

model = Gmm()
model.train("sample_data/car", synthetic_dir="synthetic/car/gmm")
```

Or via the batch runner covering all five acceptance-criteria datasets:

```bash
MODEL=gmm bash scripts/run_new_models.sh
```

(`scripts/run_new_models.sh` loops `MODEL` over `gmm`, `bayesian_network`,
`tabdiff`, `tabmt`; set `MODEL=gmm` in your shell first, or edit the `MODELS`
array, to run just this one.)

## Status

Complete and verified. Ran end-to-end (train → sample → TSTR evaluate) on all
5 datasets; results in `Results/<dataset>/gmm_tstr.csv`.
