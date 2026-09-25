# SMOTE Model

## Model Overview

Katabatic provides a SMOTE-family implementation for oversampling imbalanced tabular datasets.

The implementation supports three variants:

- `smote` for numerical features;
- `smotenc` for datasets containing both numerical and categorical features;
- `smoten` for categorical-only datasets.

The default remains standard SMOTE so existing Katabatic SMOTE benchmark scripts remain backward compatible.

SMOTE is an oversampling method rather than a standalone generative model. It increases representation of minority classes by creating additional observations from existing minority-class neighbourhoods.

---

## Research Paper

The main SMOTE implementation was reviewed against:

> N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,
> **“SMOTE: Synthetic Minority Over-sampling Technique,”**
> *Journal of Artificial Intelligence Research*, Volume 16, pp. 321–357, 2002.
> DOI: 10.1613/jair.953

The implementation should be described as **paper-aligned rather than an exact reproduction** of the original experiments.

---

## Supported Variants

## Paper Alignment Notes

The implementation was reviewed against Chawla et al. (2002).

Core behaviour aligned with the original SMOTE method:

- `k_neighbors=5` by default, matching the paper.
- Nearest neighbours are selected from the same minority class.
- For complete oversampling passes, each minority observation is used once as a source observation.
- For a partial pass, source observations are selected without replacement.
- One of the nearest minority-class neighbours is selected randomly.
- Synthetic values are generated using interpolation between the source and neighbour with a random gap between 0 and 1.

Katabatic-specific compatibility behaviour:

- Katabatic uses `sampling_strategy` rather than the paper's explicit percentage-based `N` interface.
- `k_neighbors` is reduced when a minority class is too small to support five neighbours.
- Katabatic may subsample the resampled output back to the original dataset size.
- Katabatic's benchmark and evaluation pipeline is not a direct reproduction of the classifiers and ROC/AUC experiments reported in the original paper.

## Approach
The synthetic data generation pipeline is:

### SMOTE

Use `variant="smote"` when all feature columns are numerical.

Example:

```python
from katabatic.models.smote.models import SMOTEModel

model = SMOTEModel(
    variant="smote",
    k_neighbors=5,
    sampling_strategy="auto",
    random_state=42,
)

model.train(
    data_dir=paths["split_dir"],
    synthetic_dir=paths.get("synthetic_dir"),
)

synthetic_df = model.sample(1000)
```

---

### Evaluation Pipeline Benchmark Scripts

Evaluation pipeline Benchmark Scripts for each dataset:
- Magic [benchmarks/examples/smote/magic](../../../benchmarks/examples/smote/run_smote_magic.py)
- Shuttle [benchmarks/examples/smote/shuttle](../../../benchmarks/examples/smote/run_smote_shuttle.py)

---

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). **SMOTE: Synthetic Minority Over-sampling Technique.** *Journal of Artificial Intelligence Research, 16*, 321–357. DOI: 10.1613/jair.953.
2. `imbalanced-learn` documentation for `SMOTE`, `SMOTENC`, and `SMOTEN`.
