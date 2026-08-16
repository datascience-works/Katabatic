# Bayesian Network for Katabatic

## Model Type
Probabilistic graphical model (discrete Bayesian network) for synthetic tabular
data generation.

## Model Overview
Learns a directed acyclic graph over the dataset's columns using Chow-Liu tree
structure search, fits conditional probability tables by maximum likelihood,
and generates synthetic rows via forward (ancestral) sampling from the learned
network. Numeric columns are quantile-discretized into bins before learning
and decoded back to the bin midpoint after sampling. This is the one model on
the full 62-item Katabatic model registry that no mentorship branch attempted
under this name — a genuinely unclaimed gap, not just an unmerged/broken
attempt (contrast with the CTAB-GAN+ audit finding in this same branch).

---

## Research Paper
Chow, C. K. and Liu, C. N. (1968)
Approximating Discrete Probability Distributions with Dependence Trees
IEEE Transactions on Information Theory (the Chow-Liu algorithm used for
structure learning here)

General reference: Koller, D. and Friedman, N. (2009)
Probabilistic Graphical Models: Principles and Techniques, MIT Press

---

## Implementation Details
- Structure learning: `pgmpy.estimators.TreeSearch` with the `chow-liu`
  estimator — fast and robust for higher-dimensional discrete data, unlike
  full hill-climbing search
- Parameter fitting: `pgmpy.parameter_estimator.DiscreteMLE`
- Sampling: `pgmpy.sampling.BayesianModelSampling.forward_sample`
- Numeric columns are quantile-binned (`n_bins=10` by default) via
  `utils.Discretizer`; column roles (numeric vs. categorical) are read from
  each dataset's `info.json` so integer-coded categorical columns (e.g.
  `car`) aren't mistakenly binned as if they were continuous

---

## Katabatic Model Structure

katabatic/models/bayesian_network/

- `__init__.py` → exposes `BayesianNetworkModel`
- `models.py` → structure learning, parameter fitting, sampling
- `utils.py` → quantile discretization / decode

---

## Dependencies

`pgmpy` — already a Katabatic dependency (used by `ganblr`), no new external
package. Install with:

```bash
pip install katabatic[bayesian_network]
```

---

## Dataset Format (Input)

sample_data/<dataset_name>/

- x_train.csv
- y_train.csv
- x_test.csv
- y_test.csv

## Output Format (Generated)

synthetic/<dataset_name>/bayesian_network/

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
from katabatic.models.bayesian_network.models import BayesianNetworkModel

model = BayesianNetworkModel()
model.train("sample_data/car", synthetic_dir="synthetic/car/bayesian_network")
```

Or via the batch runner:

```bash
MODEL=bayesian_network bash scripts/run_new_models.sh
```

## Status

Complete and verified. Ran end-to-end (train → sample → TSTR evaluate) on all
5 datasets; results in `Results/<dataset>/bayesian_network_tstr.csv`.

## Important Notes

- Quantile binning is lossy for numeric columns — precision is bounded by
  `n_bins`. Increase `n_bins` in `_defaults` for finer-grained numeric
  reconstruction at the cost of sparser CPTs.
- Chow-Liu structure search only learns tree-shaped dependency graphs (each
  node has at most one parent), which trades some expressiveness for
  reliability at higher dimensionality versus full structure search.
