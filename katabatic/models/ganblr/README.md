# GANBLR

GANBLR - Generative Adversarial Network modelling inspired by the relationship between Naive Bayes and Logistic Regression.
Was created to address two shortcomings of current GAN models when creating tabular synthetic data. These are the trade-off of raw predictive performance and lack of transparency around the generated raw data; creating a degradation in interpretability. Secondly, the prior knowledge is not incorporated as the focus is on raw features and not on explicit feature interactions which are not taken into account.
GANBLR, on the other hand, is designed to address these two points.

***

## Overview

This implementation integrates three core ideas from the original GANBLR paper, **adapted specifically for Katabatic**:

- **kDB feature encoding**: High-order feature interactions captured via a k-dependence Bayesian network
- **Adversarial training**: GAN-style generator/discriminator dynamics over a logistic-regression backbone
- **Discrete data focus**: Operates directly on label-encoded categorical data — no normalization or one-hot bloat

***

## Implementation Details

**Paper**: Zhang, Y., Zaidi, N. A., Zhou, J., & Li, G. (2021). *GANBLR: A Tabular Data Generation Model*. ICDM 2021. https://ieeexplore.ieee.org/document/9679177

**Repository**: Wraps the official Tulip Lab implementation https://github.com/tulip-lab/ganblr

### Paper / Library Training Recipe

```
fit(x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1)

  k:              max parents in kDB structure (paper recommends ≤ 2)
  warmup_epochs:  pre-training phase before adversarial loop
  epochs:         total training epochs
  Input:          x must be discrete (label-encoded integers)
  Loss:           cross-entropy over kDB-encoded high-order features
```

### Katabatic Implementation

**Training Loop** (via `train()`, the pipeline-facing entry point — see
[Model Contributions](../../../MODEL_CONTRIBUTIONS.md) for the shared `data_dir`/
`synthetic_dir`/`artifact_state_dir` convention every model follows):

```
Warmup ×1 epoch:   pre-train discriminator on real data
Adversarial ×N-1:  alternating generator / discriminator updates
epochs=100 (or 150 if train(..., size_category="large")), batch_size=512, k=2
```

**Data Processing (tabular-specific)**:
- Categorical → label encoding (consecutive integers 0..K-1)
- Continuous → discretized (Katabatic `discretize_preprocess`)
- Synthetic labels in tune with the real data distribution

***

## Hyperparameter Comparison

| Parameter | Paper / Library Default | **Katabatic `train()`** |
|---------|---------------|---------------|
| k | 0 (auto) | **2** |
| epochs | 10 | **100** (150 for `size_category="large"`); override via `train(..., epochs=...)` or `train(..., train_epochs=...)` |
| batch_size | 32 | **512** |
| warmup_epochs | 1 | **1** (unchanged) |
| verbose | 1 | **1** (unchanged) |

Calling `fit()` directly (rather than through `train()`) uses the paper/library defaults shown
in the left column, since `fit()` itself hasn't changed — only `train()`'s call to it overrides
`k`, `epochs`, and `batch_size`.


***

## References

- Zhang, Y., Zaidi, N. A., Zhou, J., & Li, G. (2021). *GANBLR: A Tabular Data Generation Model*. ICDM 2021. doi:10.1109/ICDM51629.2021.00103
- Zhang, Y., Zaidi, N. A., Zhou, J., & Li, G. (2022). *GANBLR++: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks*. SDM 2022. doi:10.1137/1.9781611977172.34
- Reference repository: https://github.com/tulip-lab/ganblr
- Documentation: https://ganblr-docs.readthedocs.io/en/latest/

## Generative AI Acknowledgement

AI was used to assist with interpreting and structuring the GANBLR algorithm based on the **original research paper and official repository**.

All generated content was **manually verified, modified, and extended**, including debugging, Katabatic pipeline integration, and full experimental validation. The final implementation reflects the author's independent work.
