# TabKDE Model

## Model Overview
TabKDE generates synthetic tabular data by encoding real rows into a continuous
"copula" latent space using each column's own empirical distribution (rather
than assuming a parametric distribution like Gaussian), then training a
generative model over that latent space and decoding new samples back into
realistic tabular rows.

---

### Key Idea
TabKDE avoids assuming a parametric shape (e.g. Normal) for each column's
marginal distribution. Instead, it uses the **empirical CDF** of each column
(a KDE-flavoured, non-parametric approach) to map real values to uniform
(0,1) "copula" values, which are then Gaussianised via the inverse normal CDF.
A generative model is trained on this well-behaved Gaussian latent space,
and new synthetic rows are produced by reversing the process: sample latent
→ uniform (copula) space → empirical inverse CDF → decoded real values.

---

### Research Paper

- **Paper / repo used as base:** TabKDE authors, official implementation at
  [github.com/tabkde/tabkde-main](https://github.com/tabkde/tabkde-main).
  This repository extends [TabSyn](https://github.com/amazon-science/tabsyn)
  (Zhang et al.) by adding copula/KDE-based encoding.
- **What we kept the same:** the core idea of copula/empirical-CDF encoding
  (`DataProcessor`, `EmpiricalTransformer` - ported closely from
  `tabkde/copula_encoding/model.py`) combined with a generative model trained
  on the resulting latent space.
- **What we changed:** the original repo trains an EDM-style (Karras et al.)
  diffusion model and relies on a per-dataset `info.json` plus a full
  download/preprocessing pipeline that isn't part of Katabatic. We
  re-implemented the generative stage as a standard DDPM (predict-noise,
  MSE loss, linear beta schedule) trained directly in the Gaussianised
  copula latent space, and read data via Katabatic's own
  `x_train.csv` / `y_train.csv` convention instead of `info.json`.
- **Parts not specified / inferred:** the original repo also fits a
  `gauss_model` (Gaussian mixture) into the copula space via
  `latent_utils.py`; we approximated the same idea using the standard
  Gaussian-copula trick (inverse normal CDF) rather than an explicit
  mixture model, since the mixture-fitting code wasn't available to us
  in a form compatible with Katabatic's pipeline.

---

## Approach
1. **Encode** - `DataProcessor` rank-encodes categorical columns (ordered by
   their relationship to the first principal component of numeric columns,
   when available) and standard-scales the full row.
2. **Copula transform** - `EmpiricalTransformer` fits the empirical
   (rank-based) CDF of each encoded column, producing uniform (0,1) values
   that capture each column's true distribution shape without assuming a
   parametric form.
3. **Gaussianise** - uniform values are mapped to standard-normal latents via
   the inverse normal CDF (a standard Gaussian-copula step), giving the
   generative model a well-behaved continuous space to train on.
4. **Train a generative model** - a small MLP denoiser is trained with a
   DDPM-style noise-prediction objective directly on these Gaussian latents.
5. **Sample** - reverse-diffuse from random noise → Gaussian latents →
   uniform (via normal CDF) → `EmpiricalTransformer.convert()` → encoded
   space → `DataProcessor.decode()` → real synthetic rows.

### Training Details
The denoiser is trained with Adam, MSE loss between predicted and true noise,
on batches sampled from the training set's Gaussianised latents.

### Convergence Criteria
Training stops early if the epoch loss does not improve for `patience`
consecutive epochs (default 50). In practice, on the car dataset, training
converges within roughly 100–300 epochs regardless of the configured epoch
cap (see Performance Benchmarks below).

---

## Hyperparameters
Defined in `TabKDEConfig` (`utils.py`):

| Hyperparameter | Default | Notes |
|---|---|---|
| `diffusion_epochs` | 1000 | Upper bound only - early stopping usually triggers well before this (see benchmarks) |
| `diffusion_batch_size` | 512 | |
| `hidden_dim` | 256 | Denoiser MLP hidden size |
| `diffusion_steps` | 50 | Denoising steps at sampling time |
| `lr` | 1e-3 | **Team change: 5e-4 found to perform best** (see benchmarks) |
| `weight_decay` | 0.0 | |
| `patience` | 50 | Early stopping patience |
| `seed` | 42 | |

**Recommended configuration after tuning (car dataset):** `lr=5e-4`,
`diffusion_epochs=300` (early stopping makes higher values unnecessary),
`diffusion_steps=25–50` (higher values were found to *degrade* quality -
see Performance Benchmarks).

---

## Input
Standard Katabatic convention:
- `X`: Tabular feature matrix (categorical + continuous columns, as
  specified by the caller)
- `y`: Target label column

### Expected Files
- `x_train.csv` and `y_train.csv`, **or**
- a single `train_full.csv` with the label as the last column

---

## Preprocessing

### Numerical Features
Standard-scaled (`StandardScaler`) alongside the encoded categorical columns
as part of `DataProcessor.fit()`.

### Categorical Features
Rank-encoded to integers, ordered by their relationship to the first
principal component of the numeric columns (falls back to alphabetical
category order if no numeric columns exist), then scaled like any other
numeric column.

---

## Label Handling
The label column is treated as categorical (classification target) unless
explicitly listed under `continuous_cols` by the caller. It is encoded,
trained on, and decoded identically to the feature columns, then split back
out as `y_synth` after sampling.

---

## Output
Generated files (written by `TabKDEModel.train()`):
- `x_synth.csv`
- `y_synth.csv`

(Note: unlike some other Katabatic model ports, this implementation does not
currently write a `metadata.json` - a possible follow-up improvement.)

---

## Evaluation
`TabKDEModel.evaluate()` returns a lightweight sanity-check metric: the mean
absolute difference between real and freshly-sampled synthetic data's
per-column mean/std (numeric columns only), normalised by the real column's
std. This is **not** the project's primary evaluation - full scoring is done
via the shared `SyntheticEvaluationPipeline` (fidelity, utility, diversity,
privacy, consistency, stability), as reported below.

---

## Strengths
- Fast to train - early stopping converges within ~100–300 epochs on the car
  dataset (a few seconds of wall-clock training time)
- No assumption about the shape of each column's marginal distribution
  (robust to skewed / non-Gaussian real-world columns)
- Strong fidelity (0.93) and diversity (0.99) scores after tuning - 100%
  category coverage across all columns

## Limitations
- **High exact-duplicate rate (~79–82%) across every configuration tested**,
  driving a weak privacy score (~0.45–0.47). This persisted across all
  hyperparameter, epoch, and diffusion-step configurations tested, pointing
  to a structural issue in the empirical/copula encoding on low-cardinality
  categorical datasets (like car, with only 4×4×4×3×3×3 possible feature
  combinations) rather than something fixable purely by tuning. Flagged for
  further investigation.
- Utility (downstream classifier performance) is inconsistent across
  classifiers - Logistic Regression and Linear SVM hold up reasonably well
  (TSTR close to TRTR), but tree-based models (Decision Tree, Random Forest)
  and MLP show large accuracy drops.
- In our simplified DDPM implementation, **more diffusion steps degraded
  quality rather than improving it** (composite score dropped from 0.75 at
  25–50 steps to 0.62 at 200 steps) - the opposite of typical diffusion
  model behaviour, likely due to error accumulation over many reverse steps
  with a simple fixed linear noise schedule on a small dataset.

---
## Installation

```bash
poetry add torch scikit-learn scipy
```
(torch was added as a new project dependency for this model; scikit-learn
and scipy were already present.)

---

## Usage

```python
from katabatic.models.tabkde.models import TabKDEModel

model = TabKDEModel(
    diffusion_epochs=300,
    hidden_dim=256,
    diffusion_steps=50,
    lr=5e-4,
)

model.train(
    data_dir="path_to_split_data",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
)

X_synth, y_synth = model.sample(1000), None  # or: df = model.sample(1000)
```

Evaluation pipeline benchmark script for the car dataset:
- Car: `benchmarks/examples/tabkde/run_tabkde_car.py`

---

## Model Evaluation Benchmarks Results

#### Car Dataset (best configuration: lr=5e-4, diffusion_epochs=300, diffusion_steps=50)

Composite score: **0.7520**

Dimension scores:
- fidelity 0.9259
- utility 0.7426
- diversity 0.9918
- privacy 0.4520
- consistency 0.4501
- stability 0.9715

---

## Model Performance Benchmarks Results

Baseline configuration (`diffusion_epochs=1000, hidden_dim=256,
diffusion_steps=50, lr=1e-3`) took under 10 seconds to train on CPU on the
car dataset (1382 training rows), due to early stopping.

### 1. Hyperparameter tuning (5 configurations)

| Config | lr | hidden_dim | diffusion_steps | Composite | Fidelity | Utility | Privacy |
|---|---|---|---|---|---|---|---|
| baseline | 1e-3 | 256 | 50 | 0.7117 | 0.8165 | 0.7313 | 0.4703 |
| fewer_epochs | 1e-3 | 256 | 50 | 0.6939 | 0.8829 | 0.6026 | 0.4559 |
| more_steps | 1e-3 | 256 | 200 | 0.6292 | 0.8498 | 0.4605 | 0.4587 |
| smaller_hidden | 1e-3 | 128 | 50 | 0.6506 | 0.8596 | 0.5088 | 0.4612 |
| **lower_lr (best)** | **5e-4** | 256 | 50 | **0.7520** | 0.9259 | 0.7426 | 0.4520 |

**Takeaway:** lowering the learning rate to 5e-4 gave the best result across
every dimension except privacy (which stayed essentially unchanged around
0.45–0.47 regardless of configuration - see Limitations).

### 2. Epoch tuning

| Epochs | Train time (s) | Composite |
|---|---|---|
| 100 | 8.85 | 0.7515 |
| 300 | 5.93 | 0.7520 |
| 500 | 5.44 | 0.7520 |
| 1000 | 5.21 | 0.7520 |
| 2000 | 5.27 | 0.7520 |

**Takeaway:** composite score is essentially flat regardless of the epoch
cap, because early stopping (`patience=50`) converges the model well before
higher caps are reached. 100–300 epochs is sufficient; anything higher adds
no benefit. This is a case of "we validated the defaults and ruled out a
more expensive configuration," per the team's benchmarking guide.

### 3. Diffusion steps tuning (sampling-time steps)

| Diffusion steps | Train time (s) | Sample time (s) | Composite |
|---|---|---|---|
| 10 | 14.45 | 1.00 | 0.7520 |
| **25** | 6.79 | 0.98 | **0.7520** |
| **50** | 6.84 | 0.99 | **0.7520** |
| 100 | 8.07 | 1.63 | 0.7220 |
| 200 | 7.78 | 2.25 | 0.6216 |

**Takeaway:** unlike typical diffusion models, more sampling steps **hurt**
quality here (composite dropped from 0.75 to 0.62 between 50 and 200 steps),
while also slowing down sampling. 25–50 steps is optimal for this dataset
and implementation.

---
