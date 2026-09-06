# ARF - Adversarial Random Forests

Adversarial Random Forests (ARF) is a tree-based generative
model for density estimation and synthetic tabular data generation.

ARF iteratively trains random forest classifiers to distinguish
real observations from synthetic observations. The learned forest
partitions the feature space into regions in which variables are
approximately locally independent.

The complete synthetic data generation workflow consists of:

1. ARF - learns the independence-inducing forest partitions.
2. FORDE - estimates feature distributions within the learned leaves.
3. FORGE - samples new synthetic observations from those distributions.

---

## Implementation

**Paper:** Watson, D. S., Blesch, K., Kapar, J., & Wright, M. N.
(2023). *Adversarial Random Forests for Density Estimation and
Generative Modeling*. Proceedings of AISTATS 2023, PMLR 206.

**Paper link:** https://proceedings.mlr.press/v206/watson23a.html

---

This Katabatic implementation uses the Python `arfpy` package,
which implements ARF, FORDE and FORGE.

**Repository:** https://github.com/bips-hb/arfpy/tree/master

---

## Generative AI Acknowledgement

AI was used to assist with interpreting and structuring the GANBLR algorithm based on the **original research paper and official repository**.

All generated content was **manually verified, modified, and extended**, including debugging, Katabatic pipeline integration, and full experimental validation. The final implementation reflects the author's independent work.