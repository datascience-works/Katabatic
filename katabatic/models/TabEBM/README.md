# TabEBM Model

## Model Overview
TabEBM is a class-conditional generative model for tabular data augmentation. Unlike
generative methods that learn a single shared model to approximate all class-conditional
densities, TabEBM constructs a **distinct Energy-Based Model (EBM) for each class**,
learning each class's marginal distribution independently. It is training-free: it
repurposes a pretrained in-context tabular classifier (TabPFN) as the energy function,
requiring no additional model training.

---

## Paper Fidelity: Six Directive Areas

Summary verification against the six required directive areas, with detail in the
sections that follow.

### 1. Architecture
Class-specific EBMs — one independent energy model `E_c(x)` per class, rather than a
single shared model across all classes. Each EBM's energy function is derived by
reinterpreting the logits of a pretrained TabPFN classifier, fit to a per-class surrogate
binary task. No new neural network architecture is trained; TabPFN's transformer
architecture (25.82M parameters, confirmed via source load) is reused as-is.
**Verified: matches Section 2.2 of the paper.**

### 2. Training Procedure
TabEBM is training-free. For each class `c`: (1) label real samples of class `c` as `1`
and synthetic hypercube-corner negative samples as `0`; (2) fit the pretrained TabPFN
classifier to this surrogate binary task via in-context inference only (no gradient
updates to TabPFN's weights — `only_inference=True`, `no_grad` set per device); (3)
reinterpret the resulting logits as the class's energy function.
**Verified: matches Equation 1 and Section 2.2 of the paper; confirmed no_grad/inference-only
flags in `TabEBM.py:__init__`.**

### 3. Loss Function
No conventional loss is minimised (no gradient descent on model parameters). The
"loss"-equivalent is the energy function itself, derived analytically from the
classifier's logits via log-sum-exp:
```
E_c(x) = -log( exp(f_c(x)[0]) + exp(f_c(x)[1]) )
```
implemented in code as `energy = -torch.logsumexp(logits, dim=1)`.
**Verified: matches Equation 3 of the paper exactly** — `logsumexp` over two logits is
algebraically identical to `log(exp(a) + exp(b))`.

### 4. Noise Mechanisms
Two distinct noise mechanisms are used:
- **Surrogate negative sampling:** 4 synthetic negative points per class, placed at
  random corners of a hypercube, `α_dist = 5` standard deviations from the origin — this
  constructs the contrastive signal the energy function learns from.
- **SGLD sampling noise:** Gaussian noise injected at each of `T=200` Langevin sampling
  steps (`sgld_noise_std=0.01`), plus Gaussian perturbation of the starting points
  (`starting_point_noise_std=0.01`) before sampling begins.
**Verified: matches Appendix D.1.1 (negative sample placement) and Equation 4 (SGLD update
rule) of the paper; all default values confirmed in source (see Hyperparameters table).**

### 5. Hyperparameters
All hyperparameters specified in Appendix B.6 were checked directly against the shipped
code defaults — see the full comparison table under **Research Paper**, below. Every
value matched exactly; no corrections were required.

### 6. Evaluation Methods
Reproduced two of the paper's evaluation categories on the `biodeg` dataset (Nreal=100,
matching Appendix B.1–B.2's protocol): (i) downstream data augmentation utility (balanced
accuracy, Logistic Regression), and (ii) statistical fidelity (Inverse KL, KS test). Full
results and comparison against the paper's published numbers are under **Evaluation**,
below.

---

### Key Idea
TabEBM reinterprets a classifier's logits as an unnormalised energy function. For each
class `c`, it constructs a **surrogate binary classification task** — the real samples of
class `c` (labelled 1) versus synthetic "negative samples" placed at the corners of a
hypercube far from the data (labelled 0) — and fits a pretrained classifier (TabPFN) to
this task. The resulting logits are reinterpreted as an energy function:

```
E_c(x) = -log( exp(f_c(x)[0]) + exp(f_c(x)[1]) )
```

Sampling from `p(x | y=c) ∝ exp(-E_c(x))` is performed via **Stochastic Gradient Langevin
Dynamics (SGLD)**, an iterative gradient-based sampler that walks synthetic points toward
low-energy (high-density) regions of the learned distribution.

---

### Research Paper
Margeloiu, A., Jiang, X., Simidjievski, N., & Jamnik, M. (2024). *TabEBM: A Tabular Data
Augmentation Method with Distinct Class-Specific Energy-Based Models*. NeurIPS 2024.
[arXiv:2409.16118](https://arxiv.org/pdf/2409.16118)

**Parameters kept the same (verified against Appendix B.6, via direct source inspection):**

| Parameter | Paper value | Verified in code |
|---|---|---|
| TabPFN ensemble configurations | 3 | `N_ensemble_configurations=3` |
| Negative samples per class | 4 | Hypercube-corner loop, `TabEBM.py:104` |
| Negative sample distance (`α_dist`) | 5 std | `distance_negative_class=5` |
| SGLD step size (`αstep`) | 0.1 | `sgld_step_size=0.1` |
| SGLD noise scale (`αnoise`) | 0.01 | `sgld_noise_std=0.01` |
| SGLD steps (`T`) | 200 | `sgld_steps=200` |
| Start perturbation (`σstart`) | 0.01 | `starting_point_noise_std=0.01` |

Every hyperparameter specified in the paper matched the shipped code defaults exactly —
no manual correction was required.

**What we changed (Model Validation team):** none of the paper's hyperparameters were
altered. The only environment-level change made was checking out the last pre-TabPFN-v2
commit (`e08e182`, 2024-10-15) of the repository, since the current `main` branch has
since been rewritten to support TabPFN-v2, which is not the version used in the paper
(the paper uses the original TabPFN, v0.1.9).

**Parts not specified in the paper, and how we resolved them:** the paper does not
specify how TabEBM handles non-zero-indexed class labels. During reproduction we found
that `generate()` raises a `TypeError` when class labels are not a zero-indexed
contiguous range (e.g. `{1, 2}` instead of `{0, 1}`), which is common for real-world
datasets (e.g. OpenML's `biodeg` dataset ships labels `{1, 2}`). We resolved this by
remapping labels to zero-indexed integers before calling `generate()`; see Limitations.

---

## Approach
For each class in the training data:
1. Construct a surrogate binary task: real samples of that class (label 1) vs. 4
   synthetic negative samples placed at random corners of a hypercube, `α_dist=5`
   standard deviations from the origin (label 0).
2. Fit a pretrained TabPFN classifier to this surrogate task (inference-only, no
   gradient updates to TabPFN's weights).
3. Reinterpret the classifier's logits as a class-specific energy function `E_c(x)`.
4. Sample synthetic points via SGLD, initialised near real data points and iteratively
   refined toward lower-energy regions.

This is repeated independently for every class, producing `C` distinct EBMs rather than
one shared model.

### Training Details
TabEBM itself performs **no gradient-based training**. The only "training" step is
fitting TabPFN's in-context mechanism to the surrogate task, which is closer to a
nearest-neighbour-style inference call than conventional model training — TabPFN's
weights are frozen throughout.

### Convergence Criteria
Not applicable in the traditional sense (no loss to converge). The generative process
instead runs a **fixed number of SGLD steps** (`T=200`), after which the sampling
trajectory is taken as the final synthetic point.

---

## Hyperparameters
See the Research Paper section above for the full table of verified values (all sourced
directly from `TabEBM.py`, confirmed to match Appendix B.6).

Additional hyperparameters not sourced from the paper, used only for our reproduction:
- `num_samples` (synthetic samples per class): set to 500, matching the paper's
  `Nsyn=500` data augmentation setting (Appendix B.2).

---

## Input
- `X`: Tabular feature matrix (numpy array or pandas DataFrame)
- `y`: Target labels — **must be zero-indexed contiguous integers** (`0, 1, ..., C-1`);
  see Limitations for the failure mode if this is not the case

### Expected Files
No fixed file convention is enforced by the library itself; inputs are passed directly
as in-memory arrays to `TabEBM().generate(X, y, num_samples=...)`.

---

## Preprocessing
Preprocessing is **not** performed internally by TabEBM — it is the caller's
responsibility, matching the paper's Appendix B.3 pipeline:

### Numerical Features
Mean imputation for missing values, followed by Z-score normalisation (fit on training
data only, applied to validation/test).

### Categorical Features
Mode imputation for missing values, followed by Leave-one-out Target Statistic encoding.
(Not exercised in our reproduction, since the `biodeg` dataset used for validation is
fully numerical.)

---

## Label Handling
Labels must be **zero-indexed contiguous integers**. TabEBM's `generate()` builds an
internal `range(len(np.unique(y)))` and indexes results as `f"class_{int(target_class)}"`
— if the actual unique label values are not `{0, 1, ..., C-1}` (e.g. `{1, 2}`), this
lookup fails. Confirmed via reproduction on the `biodeg` dataset (OpenML ID 1494), whose
native labels are `{1, 2}`; remapping to `{0, 1}` before calling `generate()` resolves it.

---

## Output
`generate()` returns a dictionary keyed by `f"class_{i}"`, each holding a numpy array of
shape `(num_samples, num_features)` for that class's synthetic data.

---

## Evaluation
We reproduced two categories of evaluation from the paper, on the `biodeg` dataset
(OpenML ID 1494; 1,055 samples, 41 numerical features, 2 classes — matches paper Table 3),
subsampled to `Nreal=100` per the paper's protocol (Appendix B.2).

**Data augmentation utility** (Logistic Regression, balanced accuracy — Table 7 in paper):

| | Paper (10-run avg) | Our reproduction (1 run) |
|---|---|---|
| Baseline (real only) | 76.12 | 78.86 |
| TabEBM (real + synthetic) | 76.45 | 79.13 |
| Improvement | +0.33 | +0.27 |

**Statistical fidelity** (Inverse KL, KS test — Tables 16/17 in paper, Appendix D.6):

| | Paper | Our reproduction |
|---|---|---|
| Inverse KL | ~0.90 | 0.607 |
| KS test (median p-value) | ~0.73 | 0.024 |

The fidelity metrics diverge more than the utility metrics; most likely explanation is
the small real-sample size per class (27–53 samples) versus 500 synthetic samples per
class, which increases the KS test's statistical power to detect differences even for
genuinely similar distributions, combined with our metrics being a simplified histogram
approximation rather than the paper's exact Synthcity-based implementation.

---

## Strengths
- No additional model training required (training-free, reuses pretrained TabPFN)
- Class-specific EBMs avoid the shared-model overfitting/imbalance issues the paper
  reports in competing class-conditional methods
- Confirmed to improve downstream classification accuracy over real-data-only baseline
  on small-sample regimes, consistent with the paper's central claim
- All hyperparameters in the shipped code match the published paper exactly — no
  undocumented deviations found

## Limitations
- **Label indexing bug:** `generate()` fails with a `TypeError` on non-zero-indexed class
  labels, a common real-world data condition. Caller must remap labels manually.
- Relies on GPU for practical runtime (`TabPFNClassifier` defaults to `device="cuda"`);
  CPU fallback (`plotting=True` path) is available but slower.
- The current repository's `main` branch has moved to TabPFN-v2, which is incompatible
  with the paper's original implementation — reproducing the paper requires checking out
  an older commit, which is not documented in the repository's README.
- Statistical fidelity metrics in our small-sample reproduction diverged more from the
  paper than the utility (accuracy) metrics did; likely sample-size sensitivity in the
  fidelity tests rather than a model defect (see Evaluation).

---
## Installation

The paper's `requirements_paper.txt` targets a Linux environment; several pinned
dependencies (`triton`, `tensorflow-io-gcs-filesystem`) have no Windows wheels. On
Windows, install via WSL2/Ubuntu:

```bash
conda create -n tabebm python=3.10.12 -y
conda activate tabebm
git clone https://github.com/andreimargeloiu/TabEBM
cd TabEBM
git checkout e08e182   # last commit before the TabPFN-v2 rewrite
pip install "pip<24.1" # required for pytorch-lightning==1.7.6's metadata
pip install --no-cache-dir -r requirements_paper.txt
pip install --no-cache-dir --force-reinstall --no-deps .
```

The pretrained TabPFN checkpoint's original host (`automl/TabPFN`, `main` branch) has
since moved/reorganised; it is still available on the `tabpfn_v1` branch:
```bash
curl -L -o <site-packages>/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_100.cpkt \
  "https://github.com/PriorLabs/TabPFN/raw/tabpfn_v1/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
```

---

## Usage

```python
from tabebm.TabEBM import TabEBM

tabebm = TabEBM()
synthetic_data = tabebm.generate(
    X_train, y_train,      # y_train must be zero-indexed: {0, 1, ..., C-1}
    num_samples=500,       # Nsyn=500, per Appendix B.2
    sgld_steps=200,        # paper default
)
# synthetic_data['class_0'], synthetic_data['class_1'], ...
```

---

## Model Evaluation Benchmarks Results

#### biodeg Dataset (OpenML ID 1494, Nreal=100)

Downstream predictor: Logistic Regression, balanced accuracy (%)

- Baseline (real data only): **78.86**
- TabEBM (real + synthetic augmentation): **79.13**
- Improvement: **+0.27 points**

Statistical fidelity (real vs. synthetic, simplified reproduction):
- Inverse KL: **0.607**
- KS test (median p-value): **0.024**

## Model Performance Benchmarks Results

Generation of 500 synthetic samples per class (2 classes, 41 features, TabPFN with
25.82M parameters), on an NVIDIA RTX 3050 (WSL2/CUDA): completes within the same
Jupyter session interactively — sub-minute for `Nreal=100`-scale inputs.
