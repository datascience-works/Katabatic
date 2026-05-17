# TabKDE Model

## Model Overview
TabKDE is a **copula-based kernel density estimation generative model** for tabular data that achieves high accuracy and scalability without VAEs, diffusion, or one-hot encoding.

The model maps tabular data into a unit hypercube via a copula transform, estimates a non-parametric kernel density using the distance-to-closest-record (DCR) distribution, and generates synthetic samples via boundary-aware KDE sampling.

This implementation follows the algorithm described in *TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates* and is fully self-contained using numpy, pandas, scikit-learn, and scipy.

---

## Approach
The model follows a three-step pipeline (Algorithm 12):

1. **Encode** — convert all features into a unified numerical format
2. **Map to latent space** — apply a copula transform E → Z ∈ [0, 1]^d
3. **Generate** — sample new points via boundary-aware KDE in latent space, then invert the copula

### Key Idea
By mapping data into the unit hypercube via empirical CDF transforms, all marginal distributions become uniform. Sampling in this space then becomes a matter of perturbing existing training points in a covariance-aware direction by a radius drawn from the learned DCR distribution — without requiring any parametric distributional assumption.

---

## Encoding (T → E)
Each feature type is handled separately to avoid one-hot encoding:

### Numerical Features
- Passed through unchanged

### Ordinal Features
- Mapped to consecutive integers 1, 2, 3, ... preserving natural order

### Categorical Features
- Encoded via **PrincipalGuidedEncoding** (Algorithm 5):
  - Computes the top PCA direction of numerical features
  - Assigns each category the mean projection of its rows onto that direction
  - Collapses each categorical column to a single continuous dimension

---

## Copula Transform (E → Z)
The encoded data is mapped to Z ∈ [0, 1]^d using the empirical CDF per column (Algorithm 7):

```math
z_{i,j} = \hat{F}_j(e_{i,j})
```

The sorted column values are stored to enable exact inversion. The sample covariance Σ of Z is computed to guide directional sampling.

---

## DCR Distribution (Algorithm 9)
The distance-to-closest-record distribution is estimated empirically:

- Repeatedly splits Z into two random halves
- Computes nearest-neighbour distances between splits
- Fits a **Gaussian Mixture Model** (BIC-selected, k = 1…10) to the collected distances

This GMM defines the kernel bandwidth adaptively from the data.

---

## Sampling (Algorithm 13 — SampleKDE-iterative)
For each synthetic sample:

1. Uniformly pick an anchor z_i from Z
2. Sample radius r > 0 from the GMM
3. Sample direction u ~ N(0, Σ) / ‖·‖
4. Propose z' = z_i + r · u
5. For any out-of-[0,1] coordinates, resample only those components of u (preserving scale) and re-propose
6. Retry up to `max_kde_attempts` times before restarting

This boundary-aware step ensures all generated points remain in [0, 1]^d, respecting the marginal support of each feature.

---

## Inverse Copula (Z → E, Algorithm 8)
Generated latent points are mapped back to the original feature space:

- **Numerical**: linear interpolation between two bracketing empirical values
- **Categorical / Ordinal**: probabilistic rounding — picks one of the two nearest category values, weighted by distance

---

## Hyperparameters
Defined in `TabKDEModel`:

- `n_dcr_splits = 20`
- `max_gmm_components = 10`
- `max_kde_attempts = 50`
- `ord_cols = []` (auto-detected if not supplied)
- `cat_cols = []` (auto-detected if not supplied)
- `seed = 42`

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

---

## Label Handling
Labels are **not** passed through the KDE generative process. Instead:
- `y` is stored from training
- Synthetic labels are sampled empirically from the training label distribution

This guarantees all classes appear in correct proportions regardless of rarity, and avoids the label column distorting the copula transform.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`

---

## Evaluation
`evaluate()` returns the **mean column-wise KS statistic** between real and synthetic numeric features. Lower is better; 0 indicates identical marginal distributions.

---

## Strengths
- No neural network training — extremely fast (seconds to minutes)
- Handles mixed numerical, ordinal, and categorical features without one-hot encoding
- Principled boundary control via iterative unit-hypercube enforcement
- Scalable to large datasets and high category counts
- Adaptive kernel bandwidth learned from data via GMM-DCR

---

## Limitations
⚠️ The model may struggle with:
- Datasets with no numerical features (PCA direction for PrincipalGuidedEncoding cannot be computed)
- Very high-dimensional datasets where boundary rejection becomes frequent
- Highly multimodal distributions where a single KDE pass undersamples modes

⚠️ Uses:
- Empirical CDF inversion (quantile interpolation) for numerical reconstruction
- Probabilistic rounding for categorical reconstruction

These may introduce minor discretisation artefacts on categorical-heavy datasets.

---

## Usage

```python
from models import TabKDEModel

model = TabKDEModel()

model.train(
    output_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```
