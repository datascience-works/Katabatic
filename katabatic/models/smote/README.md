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
