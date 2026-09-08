# ForestDiffusion Model

## Model Overview
ForestDiffusion is a tabular synthetic data generator that replaces the neural-network score/vector-field estimator used in typical diffusion or flow-matching models with **Gradient-Boosted Trees (XGBoost by default)**. Rather than training one neural network to estimate the score function or vector field at all noise levels, this implementation trains a *separate GBT regressor for every (class label, noise level) pair*, then generates new rows by simulating the reverse-time process with those regressors standing in for the score/vector-field function. It is a Katabatic wrapper (`ForestDiffusionModel`) around a self-contained implementation supporting **both** of the paper's two generative branches — VP-diffusion and conditional flow matching.

---

### Key Idea
The model follows the general score/flow-based generative modelling recipe: perturb real data at a sequence of levels `t ∈ [0, 1]` (forward process), learn to estimate either the score function `∇x log p_t(x)` (VP-diffusion branch) or the vector field connecting noise to data (flow-matching branch) at each level, then generate new samples by starting from pure Gaussian noise and simulating the reverse-time process with Euler (flow) or Euler–Maruyama (VP) integration. The distinguishing idea of the source paper is to use **Gradient-Boosted Trees instead of neural networks** as the function approximator, motivated by GBTs' strong track record on tabular data and their ability to handle missing values natively during splitting.

---

### Research Paper
This implementation is based on: Jolicoeur-Martineau, Fatras & Kachman, *Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees*, AISTATS 2024 (arXiv:2309.09968), official code at [`SamsungSAILMontreal/ForestDiffusion`](https://github.com/SamsungSAILMontreal/ForestDiffusion).

**What was kept the same**
- The core idea of training one GBT (XGBoost by default) per noise level rather than a single neural network, with training parallelized over multiple CPU cores via `joblib.Parallel(n_jobs=...)` instead of requiring a GPU.
- **Both of the paper's generative branches are implemented**: VP-diffusion (`diffusion_type="vp"`) and conditional flow matching (`diffusion_type="flow"`, the default) — matching the paper's own recommendation that flow matching is generally the faster, better-performing option.
- Training a **separate set of models per class label** rather than conditioning on the label as an input feature, matching the paper's Section 3.8 approach.
- `n_t = 50` noise levels and `duplicate_K = 100` row-duplication as defaults, matching the paper's own ablation finding that `n_t = 50` is sufficient for competitive performance and that a high `duplicate_K` (number of independent noise draws per real row) is one of the most important hyperparameters for generation quality.
- No L1/L2 regularization and `n_estimators = 100` — matches the paper's stated principle that, since the goal is estimating a score/vector-field function rather than a predictive model, overfitting is of little concern while underfitting is the main risk, so regularization is left off and other XGBoost hyperparameters at their defaults.
- **Categorical features are one-hot (dummy) encoded**, matching the paper's stated preprocessing exactly, with a corresponding inverse transform (`decode_categorical_onehot`) mapping generated one-hot blocks back to the original category labels.
- Support for swapping the underlying GBT (XGBoost / LightGBM / CatBoost / Random Forest) mirrors the paper's own ablation across tree methods, which concluded XGBoost performs best — consistent with this implementation defaulting to XGBoost.

**What was changed / narrowed (Model Validation team should be aware of):**
- **REPAINT is not implemented.** `impute()` implements only the paper's base imputation loop (matching observed values back in at each reverse step); it does not implement the paper's REPAINT extension (Algorithm 4), which periodically jumps the trajectory backward and re-runs a few steps for better consistency between imputed and observed regions. This is called out directly in the method's own docstring as a known follow-up.
- **`evaluate()` uses a simplified proxy metric.** It computes the mean per-column Wasserstein distance between a freshly generated batch and the real training data, computed in the model's internal min-max-scaled representation. The paper's own Wtrain/Wtest metric is computed on a Gower-distance representation of the *joint* mixed-type distribution, not independent per-column marginals — this implementation's metric is a coarser, marginals-only proxy, as noted in the method's own docstring.
- **Labels are not jointly generated with features.** Synthetic labels are drawn independently from the empirical class-frequency distribution (`np.random.multinomial`) rather than being decoded from the same reverse-diffusion trajectory as the features, so `X_synth` and `y_synth` are not jointly modelled.

**Not specified in the paper / inferred by the Model Validation team:**
- `max_depth = 7`, `n_estimators = 100`, and `seed = 666` match the official library's own published defaults (confirmed against the package), so despite not being stated in the paper text itself, these are not a deviation.
- `beta_min = 0.1` / `beta_max = 20.0` for the VP noise schedule matches the paper's appendix (Algorithms 5–7), which uses `β = [0.1, 20]` throughout its worked forward/reverse examples.
- `int_indexes` refers to column positions in the model's **internal, post-encoding** layout (numeric columns first, then one-hot categorical blocks), not the original dataframe's column order — any caller building this list from the raw input columns should account for the reordering, or it will silently round the wrong columns.

---

## Approach
1. Load `x_train.csv` / `y_train.csv`, auto-detect categorical columns by dtype and one-hot encode them (numeric columns kept first, one-hot blocks appended after), and drop rows that are entirely missing.
2. Min-max scale all features to `[-1, 1]` (matching the paper's normalization step) and split rows by class label.
3. For each class label and each of `n_t` noise levels `t`, apply the forward process (VP-SDE noise injection, or the linear noise→data interpolation for flow matching) to `duplicate_K` tiled copies of that class's rows, and fit one GBT regressor per `(class, t)` pair to predict the corresponding score/vector-field target.
4. To generate, draw a batch from the standard normal prior and run the reverse-time process — Euler integration for flow matching, Euler–Maruyama for VP-diffusion — querying the appropriate per-class, class-probability-weighted regressor ensemble at each step, then undo the min-max scaling, clip to the observed training range, and decode one-hot categorical blocks back to their original labels.
5. Sample synthetic labels independently from the empirical class-frequency distribution and concatenate with the generated features.

### Training Details
No mini-batch SGD is used (as in the paper — GBTs are fit on the full duplicated dataset at once per `(class, timestep)` pair). For each pair, `duplicate_K` tiled copies of that class's data are noised and a fresh regressor (XGBoost by default) is fit to the forward-process target with `n_estimators=100` boosting rounds and no L1/L2 penalty. All `n_t × n_classes` regressors are trained via `joblib.Parallel(n_jobs=self.n_jobs)`, matching the official library's CPU-parallel training approach (default `n_jobs=-1` uses all available cores).

### Convergence Criteria
Each individual XGBoost regressor "converges" by simply exhausting its fixed `n_estimators` boosting rounds — there is no early stopping, validation set, or held-out loss monitored anywhere in training. At the model level there is no iterative training loop or convergence criterion at all: each of the `n_t × n_classes` regressors is fit exactly once.

---

## Hyperparameters
From `ForestDiffusionModel.__init__`:

| Hyperparameter | Default | Description |
|---|---|---|
| `n_t` | 50 | Number of noise levels / regressors per class |
| `model` | "xgboost" | GBT backend: `xgboost`, `random_forest`, `lgbm`, or `catboost` |
| `diffusion_type` | "flow" | Generative branch: `"flow"` (conditional flow matching) or `"vp"` (VP-diffusion) |
| `duplicate_K` | 100 | Number of noise draws per training row per timestep — a key fidelity hyperparameter per the paper |
| `max_depth` | 7 | Max tree depth (XGBoost/CatBoost) |
| `n_estimators` | 100 | Number of trees per regressor |
| `eta` | 0.3 | XGBoost learning rate |
| `tree_method` | "hist" | XGBoost tree construction algorithm |
| `reg_alpha` / `reg_lambda` | 0.0 / 0.0 | L1 / L2 regularization (off, per the paper) |
| `subsample` | 1.0 | Row subsampling fraction per tree |
| `num_leaves` | 31 | LightGBM leaf count (unused for other backends) |
| `int_indexes` | [] | Post-encoding column positions to round to integers after generation (see *Preprocessing*) |
| `beta_min` / `beta_max` | 0.1 / 20.0 | VP-SDE noise schedule bounds (used only when `diffusion_type="vp"`) |
| `eps` | 1e-3 | Minimum process time used during reverse sampling |
| `n_jobs` | -1 | CPU parallelism for training the `n_t × n_classes` regressors |
| `gpu_hist` | False | Use GPU histogram training for XGBoost |
| `seed` | 666 | Random seed (matches the paper's own choice of seed) |

`n_t`, `duplicate_K`, `diffusion_type`, `beta_min`/`beta_max`, and the choice of GBT `model` are the parameters most directly tied to the paper's formulation and are the ones Model Validation should prioritize when auditing fidelity; `max_depth`/`eta`/`tree_method`/`subsample` are standard XGBoost knobs not specifically discussed in the paper.

---

## Input
Standard Katabatic convention:
- `X`: Tabular feature matrix
- `y`: Target labels (categorical or continuous)

### Expected Files
- `x_train.csv`
- `y_train.csv`

Unlike some other Katabatic models, this wrapper does **not** support a single combined `train_full.csv` — `train()` unconditionally reads `x_train.csv` and `y_train.csv` from `output_dir` and will raise a `FileNotFoundError` (via pandas) if either is absent. Also note the entry-point parameter is named `output_dir` rather than the `data_dir` convention used elsewhere in the codebase (e.g. the Flow-VAE model) — a naming inconsistency worth flagging for repository-health cleanup.

---

## Preprocessing
- **Numerical features:** min-max scaled to `[-1, 1]` via `sklearn.preprocessing.MinMaxScaler`, matching the paper's normalization step.
- **Categorical features:** columns are auto-detected by dtype (anything not numeric, via `is_numeric_dtype`) and one-hot encoded with `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")`, matching the paper's stated dummy-encoding preprocessing. There is no manual `cat_indexes` parameter — detection is automatic rather than caller-specified.
- Integer-valued columns can optionally be flagged via `int_indexes` for post-generation rounding (see `_clip`); these indices refer to the post-encoding (numeric-first, then one-hot) column layout, not the original dataframe order.

---

## Label Handling
If a categorical/discrete label column is provided, the model trains a **fully separate list of `n_t` regressors per class** (looping over `self.y_vals`), matching the paper's Section 3.8 approach, rather than conditioning a single model on the label as an extra feature. At generation time, synthetic labels are **not** produced by the reverse process itself — they are drawn independently from the empirical class-frequency distribution (`np.random.multinomial`) and concatenated onto the generated features. This mirrors the same "generate features and labels independently" pattern seen in the Flow-VAE model in this repo, so `X_synth` and `y_synth` are not jointly modelled here either.

---

## Output
- `x_synth.csv`
- `y_synth.csv`

No `metadata.json` is written by this model (unlike Flow-VAE), so none of the training configuration, noise schedule, or per-model details are persisted alongside the synthetic data. Categorical columns in `x_synth.csv` are correctly decoded back to their original category labels via `decode_categorical_onehot` before being written out.

---

## Evaluation
`evaluate()` computes the **mean per-column Wasserstein (earth-mover's) distance** between a freshly generated synthetic batch and the real training data, computed in the model's internal min-max-scaled representation (via `scipy.stats.wasserstein_distance`); lower is better, with `0` indicating identical marginal distributions. This is a simple, single-number distributional-closeness check and is not the paper's own Wtrain/Wtest metric (Table 1 in the paper), which is computed on a Gower-scaled representation of the *joint* mixed-type distribution — this implementation's metric only checks per-column marginals, not diversity, downstream utility, or joint structure, as the method's own docstring notes.

---

## Strengths
- Trains on CPU only, no GPU required — the paper's headline advantage, since GBTs (unlike NNs) don't need dedicated hardware.
- Implements **both** of the paper's generative branches (VP-diffusion and flow matching), with flow matching — the paper's generally better-performing variant — as the default.
- Correctly implements the paper's key duplication hyperparameter (`duplicate_K`) and dummy/one-hot categorical encoding, both called out in the paper as important for generation fidelity.
- Supports swapping the GBT backend (XGBoost/LightGBM/CatBoost/RandomForest) without changing any other code.
- Per-class regressors mean the label distribution is respected exactly at sample time (frequency-matched sampling), and the paper's own ablations found per-class models help fidelity.
- Likelihood-free but principled score/flow-based generative approach, avoiding GAN-style adversarial training instability.
- Training is parallelized across CPU cores via `joblib`, matching the official library's approach.

## Limitations
- No REPAINT-style imputation (Algorithm 4) — `impute()` implements only the base reverse-process imputation loop, without the periodic backward-jump refinement the paper describes for better consistency between imputed and observed values.
- `X_synth` and `y_synth` are generated independently (labels are frequency-sampled, not decoded from the same reverse trajectory as the features), the same joint-modelling gap noted in the Flow-VAE model's README.
- `evaluate()`'s per-column Wasserstein distance is a coarser proxy for the paper's joint Gower-distance-based Wtrain/Wtest metric — useful as a quick sanity check, not a substitute for the paper's full evaluation protocol.
- No `metadata.json` is written alongside synthetic outputs, so training configuration isn't persisted for later inspection.
- The entry-point parameter name `output_dir` is inconsistent with the `data_dir` convention used elsewhere in the codebase.
- `int_indexes` must be specified in post-encoding column order, which is easy to get wrong if built from the original dataframe's column order.

---
## Installation

poetry install --extras codi

---

## Usage
```python
from models import ForestDiffusionModel

model = ForestDiffusionModel(
    n_t=50,
    model="xgboost",
    diffusion_type="flow",
    duplicate_K=100,
    max_depth=7,
    n_estimators=100,
    eta=0.3,
)

model.train(
    output_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```

Evaluation pipeline Benchmark Scripts for each dataset:
- Adult [benchmarks/examples/FORESTDIFFUSION/FILENAME](benchmarks/examples/FORESTDIFFUSION)
- Bank [benchmarks/examples/FORESTDIFFUSION/FILENAME](benchmarks/examples/FORESTDIFFUSION)
- Car [benchmarks/examples/FORESTDIFFUSION/FILENAME](benchmarks/examples/FORESTDIFFUSION)
- Credit Card [benchmarks/examples/FORESTDIFFUSION/FILENAME](benchmarks/examples/FORESTDIFFUSION)

---

## Comparison to a Related Implementation

Because this model reproduces a specific published method, the most useful comparison point is the **paper's own official implementation**, [`SamsungSAILMontreal/ForestDiffusion`](https://github.com/SamsungSAILMontreal/ForestDiffusion) (AISTATS 2024, pip-installable as `ForestDiffusion`).

**Where the two implementations line up:**
- Both center on training one XGBoost regressor per noise level (and, optionally, per class), rather than a neural network, to approximate the score/vector field — the paper's core contribution.
- Both implement the paper's two generative branches (VP-diffusion and conditional flow matching), with flow matching as the recommended default in each.
- Default hyperparameters agree closely: `n_t=50`, `duplicate_K=100`, `max_depth=7`, `n_estimators=100`, `beta=[0.1, 20]`, no L1/L2 regularization, `seed=666` all match the official library's own documented defaults.
- Both one-hot (dummy) encode categorical features, matching the paper's Section 3.6, with a proper inverse-transform back to category labels at generation time.
- Both support swapping in Random Forest / LightGBM / CatBoost as the underlying tree model alongside the default XGBoost, and both parallelize training across CPU cores.

**Where they diverge, in order of likely fidelity impact:**
1. **No REPAINT-based imputation refinement.** The official library's `impute()` supports a `repaint=True` mode (periodic backward jumps and re-runs, per the paper's Algorithm 4) for better consistency between imputed and observed regions. This wrapper's `impute()` implements only the base loop — a real but self-documented gap, and one that only affects the imputation use case, not generation.
2. **Evaluation metric fidelity.** The official library's benchmarking uses the paper's Gower-distance-based Wtrain/Wtest metric on the joint mixed-type distribution; this wrapper's `evaluate()` uses a simpler mean per-column Wasserstein distance as a lighter-weight proxy.
3. **No `p_in_one` / multi-output training optimization.** The official library added a `p_in_one=True` option to train one model across all predictors at once, purely for speed; this wrapper always trains one model per `(class, timestep)` pair with no equivalent speed optimization, though this is a performance rather than fidelity concern.
4. **Independent label sampling.** As in the official library's simpler generation mode, labels here are frequency-sampled rather than jointly diffused with features — consistent with one of the official library's supported modes, but worth noting Model Validation shouldn't expect joint label/feature coherence beyond frequency-matching.

**Summary:** the current implementation is a close, largely faithful reproduction of the paper's Forest-Flow/Forest-VP methods — both generative branches, the key `duplicate_K` hyperparameter, and dummy/one-hot categorical encoding are all correctly implemented and matched against the paper's own defaults. The remaining gaps are narrower and mostly self-documented in the code: REPAINT's imputation refinement is not implemented, and the evaluation metric is a simpler proxy for the paper's joint-distribution metric. These should be the audit's top priorities going forward, ahead of the lower-severity, cosmetic-adjacent items (metadata logging, `output_dir` naming, independent label sampling).

---

## Model Evaluation Benchmarks Results

*Not yet run for this model — populate once the shared Katabatic benchmarking harness has been executed on Adult / Bank / Car / Credit Card.*

## Model Performance Benchmarks Results

*Runtime/memory figures (e.g. "X mins on CPU with Y GB RAM") to be filled in after a benchmark pass; not available from the current code alone. Note that with `duplicate_K=100` now active by default, training will use noticeably more memory and time than a version without row duplication would — this is expected and matches the paper's own fidelity/cost tradeoff, not a regression.*
