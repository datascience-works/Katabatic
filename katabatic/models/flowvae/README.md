# Flow-VAE Model

## Model Overview
Flow-VAE is a tabular synthetic data generator that augments a standard Variational Autoencoder (VAE) with a **normalizing flow** applied to the latent code. The flow transforms the simple diagonal-Gaussian posterior produced by the encoder into a richer, more flexible posterior before decoding, which is intended to improve the fidelity of the learned latent distribution relative to a vanilla VAE. The Katabatic wrapper (`FlowVAEModel`) trains this network on a joined feature+label dataframe and exposes it through the standard Katabatic `train()` / `sample()` / `evaluate()` interface.

---

### Key Idea
The model is built on **normalizing flows for variational inference**: instead of restricting the approximate posterior `q(z|x)` to a diagonal Gaussian, a sequence of `K` invertible transformations is applied to a Gaussian sample `z_0`, producing `z_K`. Because each transformation is invertible with a tractable Jacobian determinant, the change-of-variables formula lets the model compute an exact log-density for the transformed posterior, which is used to correct the standard ELBO. The implementation supports four flow families, matching four different ways of satisfying the "tractable Jacobian" requirement:

- **Planar flow** – contracts/expands space along a hyperplane; log-det via the matrix determinant lemma.
- **Radial flow** – contracts/expands space radially around a learned reference point.
- **Householder flow** – a volume-preserving orthogonal reflection (log-det = 0 by construction).
- **NICE flow** – a volume-preserving additive coupling layer, with an optional final learned diagonal scaling layer.

The loss combines MSE reconstruction, the KL divergence term of the base Gaussian, and the flow's log-determinant correction: `loss = recon_MSE + KL(mean, log_var) − log_det`.

---

### Research Paper
This implementation is derived primarily from the normalizing-flow VAE line of work rather than a single tabular-data paper:

- Planar and radial flows follow Rezende & Mohamed (ICML 2015), *Variational Inference with Normalizing Flows*, which constructs approximate posteriors through a normalizing flow — transforming a simple initial density into a more complex one via a sequence of invertible transformations.
- Householder flow follows Tomczak & Welling (NIPS 2016 workshop), *Improving Variational Auto-Encoders using Householder Flow*, a volume-preserving orthogonal-reflection flow.
- NICE flow follows Dinh, Krueger & Bengio, *NICE: Non-linear Independent Components Estimation* (ICLR 2015 workshop), an additive coupling-layer flow.

**What was kept the same:** the closed-form Jacobian/log-determinant expressions for each flow type — verified directly against each paper's formulas — and the choice to make flow parameters **static** (learned once per model, not predicted per-input by the encoder). This is the same "static flow" design used in the source papers' own reference implementations, where flow parameters are treated as learned parameters that remain fixed with respect to the input; it significantly constrains the richness of approximate posteriors compared to input-conditioned ("dynamic") flow parameters, but is more practical and avoids the numerical instability dynamic flows can introduce.

**What was changed:**
- The base architecture was moved from image data (MNIST/CIFAR-style conv or dense encoders) to **mixed-type tabular rows**, with a fully-connected encoder/decoder instead of convolutional layers.
- A `gate` option (a sigmoid-gated linear layer, `GatedLayer`) was added to the encoder/decoder MLPs as an alternative to the plain ReLU-MLP used in the reference implementations — this is not present in any of the source papers.
- The label column is concatenated into the training matrix and reconstructed jointly with the features, but at sampling time the model's own decoded label is **discarded** and replaced with an independently sampled label drawn from the empirical training-label frequency distribution (see *Label Handling* below). This is a Katabatic-specific pipeline decision, not part of the original flow-VAE formulation.

**Inferred / not specified in the paper:**
- Concrete hidden-layer width, depth, and latent dimensionality for tabular use (the papers target image benchmarks with different scales).
- The numeric-vs-categorical column detection rule (a column is treated as numeric only if it is numeric-typed **and** has more than 10 unique values; otherwise it is one-hot encoded) — this preprocessing convention is a Katabatic addition with no equivalent in the flow papers.
- Convergence behaviour is fixed-epoch rather than criterion-based (see below), which the papers do not prescribe.

---

## Approach
1. Load and concatenate `x_train`/`y_train` (or `train_full.csv`) into a single dataframe.
2. Fit a per-column encoder (`fit_transform_tabular`): z-score numeric columns, one-hot encode categorical/low-cardinality columns (including the label), and store the resulting `TabularSchema` for later inversion.
3. Train a `FlowVAE`: encoder MLP → `(mean, log_var)` → reparameterize with a Gaussian sample → pass through the selected normalizing flow → decoder MLP reconstructs the encoded row.
4. After training, sample `n_train` rows from the standard normal prior, decode them, and invert the encoding back to original column types/scales (`inverse_transform_tabular`).
5. Save `x_synth.csv`, `y_synth.csv`, and `metadata.json` (full schema + training config + loss history).

The model learns the joint distribution of the encoded feature space by minimizing a flow-corrected ELBO — it does not use an adversarial or score-based objective.

### Training Details
Standard mini-batch SGD training with the Adam optimizer (`lr` configurable, default `1e-3`). For each epoch, the full `DataLoader` is iterated once (`shuffle=True`, `drop_last=False`); per-batch loss is `reconstruction MSE + KL divergence − flow log-determinant`, and the mean epoch loss is logged and stored in `loss_history`. Weights are initialized via each flow module's own parameter initialization (small-scale `torch.randn * 0.01`), and `random_state` seeds `random`, `numpy`, and `torch` (plus CUDA) for reproducibility.

### Convergence Criteria
There is no adaptive stopping rule, validation-loss monitoring, or early stopping. Training always runs for a **fixed number of epochs** (`epochs`, default `50`); the only "convergence" signal is the trend of `loss_history` inspected after the fact.

---

## Hyperparameters
From `FlowVAEModel.__init__`:

| Hyperparameter | Default | Description |
|---|---|---|
| `hidden_dim` | 128 | Width of encoder/decoder MLP hidden layers |
| `latent_dim` | 16 | Dimensionality of the latent code (must be even if `flow_type="nice"`) |
| `layers` | 2 | Number of hidden layers in encoder and decoder |
| `gate` | False | Use gated linear layers instead of plain ReLU MLP layers |
| `flow_type` | "planar" | One of `planar`, `radial`, `householder`, `nice`, or `none` |
| `flow_length` | 2 | Number of stacked flow transformations |
| `batch_size` | 128 | Mini-batch size (also used to chunk sampling) |
| `epochs` | 50 | Fixed number of training epochs |
| `learning_rate` | 1e-3 | Adam learning rate |
| `random_state` | 42 | Seed for `random`/`numpy`/`torch`/CUDA |
| `device` | auto | `"cuda"` if available, else `"cpu"`, unless overridden |

`flow_type`/`flow_length` and `gate` are the parameters most relevant to Model Validation, since they directly control which paper's flow formulation is active and how many transformations are stacked (paper-referenced hyperparameters); `hidden_dim`, `latent_dim`, and `layers` are dataset-scaling hyperparameters not fixed by any of the source papers.

---

## Input
Standard Katabatic convention:
- `X`: Tabular feature matrix (mixed numeric/categorical)
- `y`: Target labels (single column)

### Expected Files
- `x_train.csv`
- `y_train.csv`

Or alternatively a single `train_full.csv` with the label as the last column. `_load_training_dataframe` checks for `train_full.csv` first, then falls back to the `x_train.csv`/`y_train.csv` pair, and raises `FileNotFoundError` if neither is found. `y_train.csv` must contain exactly one column.

---

## Preprocessing
Handled entirely by `fit_transform_tabular` / `inverse_transform_tabular` in `utils.py`, applied to **every** column including the label.

### Numerical Features
A column is treated as numeric if it is a numeric dtype **and** has more than 10 unique non-null values. Numeric columns are standardized: `(x − mean) / std` (mean imputation for missing values, `std=1.0` fallback if the computed std is 0 or NaN). At inverse-transform time, the same stored `mean`/`std` are used to de-standardize.

### Categorical Features
Any column that is not "numeric" by the above rule (including low-cardinality numeric columns, e.g. binary flags) is treated as categorical: missing values are filled with the literal string `"__MISSING__"`, then one-hot encoded with `col__category` naming. The sorted category list is stored in the schema so that generation always reindexes to the same fixed set of dummy columns. Decoding takes an `argmax` over each column's one-hot block to recover the most likely category.

---

## Label Handling
The label column is **not treated specially during training** — it is encoded (numeric or categorical, per the same rule above) and concatenated into the same matrix as the features, so the FlowVAE reconstructs it jointly with `X` as part of the ELBO. However, at generation time the decoded label produced by the model is **discarded**: `y_synth` is instead generated independently by sampling from the empirical class-frequency distribution of the real training labels (with a guarantee that every class appears at least once when `n ≥` number of classes). This means the model does not currently produce features and labels that are jointly conditioned at sampling time — `X_synth` and `y_synth` are statistically independent draws. See *Limitations* below for an additional row-alignment issue this "guarantee every class appears" step introduces.

---

## Output
- `x_synth.csv`
- `y_synth.csv`
- `metadata.json` — contains the model name, the full `TabularSchema` (columns, numeric/categorical splits, means/stds/categories), and the training configuration (all hyperparameters plus the per-epoch `loss_history`).

---

## Evaluation
`evaluate()` returns the **final training-loss value** (`loss_history[-1]`, or `0.0` if training hasn't produced any recorded losses) as a single float. It does not currently compute held-out reconstruction error, likelihood, or any of the Katabatic fidelity/utility/diversity/privacy/consistency/stability dimension scores — those would need to come from the shared Katabatic benchmarking harness, not from this model class directly.

---

## Strengths
- Applicable to arbitrary mixed numeric/categorical tabular data via the shared `TabularSchema` encoder.
- The choice of flow family (`planar`/`radial`/`householder`/`nice`) lets the Model Validation team trade off expressiveness vs. numerical stability without touching the encoder/decoder code.
- Likelihood-based (ELBO) training tends to be more stable to train than adversarial approaches (no discriminator, no mode collapse dynamics).
- Deterministic, reproducible training given a fixed `random_state`.

## Limitations
- Fixed-epoch training with no early stopping or validation-based convergence check, so under/over-training is easy to miss.
- Static (input-independent) flow parameters bound the richness of the posterior. This is the same limitation the source papers' own reference implementation flags: static flows significantly constrain the richness of approximate posteriors compared to input-conditioned flow parameters, and sampling can become problematic since most latent codes can lie outside the most probable region under the standard normal prior — which may translate into lower-quality or less diverse synthetic rows.
- `X_synth` and `y_synth` are generated independently (label is frequency-sampled, not model-decoded), so any real feature–label relationship the model might have captured internally is not currently exposed in the synthetic output — this is likely to be a first-order fidelity gap.
- **Row-misalignment bug in the forced-class-representation step.** In `train()`, after `y_synth_values` is frequency-sampled, the block that guarantees every class appears at least once overwrites the first `len(label_counts)` entries of `y_synth_values` and then shuffles *only* `y_synth_values` — `x_synth` is never re-ordered to match. As a result, the forced-representation label rows end up paired with whatever `x_synth` rows happened to occupy those first positions, which can produce incoherent (feature, label) pairs beyond the independence already noted above. Worth a direct fix (shuffle `x_synth` and `y_synth` together, or forego the forced-representation step) before relying on per-class synthetic row counts for anything downstream.
- The 10-unique-value numeric/categorical cutoff is a heuristic that can misclassify low-cardinality numeric features (e.g. a 1–10 numeric score) as categorical, or high-cardinality categorical codes as numeric.
- No conditional generation (e.g. conditioning on class) is supported.

---
## Installation

poetry install --extras codi

---

## Usage
```python
from models import FlowVAEModel

model = FlowVAEModel(
    hidden_dim=128,
    latent_dim=16,
    layers=2,
    gate=False,
    flow_type="planar",
    flow_length=2,
    batch_size=128,
    epochs=50,
    learning_rate=1e-3,
)

model.train(
    data_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```

Evaluation pipeline Benchmark Scripts for each dataset:
- Adult [benchmarks/examples/FLOWVAE/FILENAME](benchmarks/examples/FLOWVAE)
- Bank [benchmarks/examples/FLOWVAE/FILENAME](benchmarks/examples/FLOWVAE)
- Car [benchmarks/examples/FLOWVAE/FILENAME](benchmarks/examples/FLOWVAE)
- Credit Card [benchmarks/examples/FLOWVAE/FILENAME](benchmarks/examples/FLOWVAE)

---

## Comparison to a Related Flow-VAE Implementation

To sanity-check the fidelity of this implementation's flow mechanics, it's useful to compare it against a public reference implementation covering the same set of flows: **[`fmu2/flow-VAE`](https://github.com/fmu2/flow-VAE)**, a PyTorch reproduction of the Rezende & Mohamed (2015) normalizing-flow VAE together with Householder and NICE flows, evaluated on MNIST/Fashion-MNIST/SVHN/CIFAR-10.

**Where the two implementations line up:**
- Both implement exactly the same four flow families — planar, radial, Householder, and NICE — with planar and radial presented as the general normalizing flows and Householder and NICE presented as volume-preserving flows, matching the `Flow` dispatch logic in `utils.py`.
- Both use the **static-flow** variant of the architecture — flow parameters treated as learned parameters that stay fixed with respect to the input — rather than having the encoder predict per-sample flow parameters (the "dynamic flow" variant the reference repo also implements but reports as impractical).
- Both use dense encoder/decoder MLPs with a configurable hidden width, depth, and latent dimensionality, and Adam as the optimizer.

**Where they diverge:**
- **Data modality.** The reference implementation targets fixed-size image tensors with convolutional or dense layers over a homogeneous pixel space; the Katabatic model targets heterogeneous tabular rows via a schema-driven numeric/categorical encoder. This is an expected and appropriate adaptation, not a fidelity gap.
- **No convolutional variant.** The reference repo also ships a convolutional static-flow VAE (`static_flow_conv_vae.py`); Katabatic's tabular setting has no image-grid structure, so this variant has no analogue here — expected.
- **Sampling caveat carried over.** The reference implementation explicitly flags that sampling becomes problematic when static flows are used, since most latent codes lie outside of the most probable region under the standard normal prior. Because Katabatic's `FlowVAE.sample()` also draws directly from a standard normal prior and decodes without any correction for this mismatch, the same failure mode is plausible here and is a good candidate for a targeted ablation (e.g. comparing sample quality with `flow_type="none"` vs. an active flow) in the divergence log.
- **Gated layers.** Katabatic's optional `gate` (sigmoid-gated linear layers) has no counterpart in the reference implementation or in the original flow papers — worth flagging as a deliberate Model Validation team addition rather than something inherited from the literature.
- **Label/target handling.** The reference implementation is unsupervised (image reconstruction only; labels are left untouched entirely). Katabatic's joint-encode-then-discard-and-resample-by-frequency approach to `y` has no equivalent to benchmark against and is purely a Katabatic-pipeline design choice (see *Label Handling* above) — including the row-misalignment bug noted in *Limitations*, which has no counterpart to compare against either since the reference implementation never touches labels at all.

**Summary:** the core flow mechanics (log-determinant formulas, static parametrization, four supported flow types) match a credible independent PyTorch implementation of the same underlying papers, which supports the correctness of the flow layer itself. The main implementation-specific risks are outside the flow layer proper — in the tabular preprocessing heuristics, the fixed-epoch convergence rule, the decoupled label-sampling strategy, and the label row-alignment bug — and are the areas this fidelity audit should prioritize.

---

## Model Evaluation Benchmarks Results

*Not yet run for this model*

## Model Performance Benchmarks Results

*Runtime/memory figures (e.g. "X mins on CPU with Y GB RAM") to be filled in after a benchmark pass; not available from the current code alone.*
