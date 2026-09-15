# ARF Model

## Model Overview
ARF (Adversarial Random Forests) is a **non-parametric generative model** for tabular data that uses an adversarial training loop to learn the joint distribution of features without assuming any parametric form.

This implementation is **fully self-contained** using scikit-learn, numpy, and pandas — no external `arfpy` dependency is required.

---

## Approach
The model follows an iterative adversarial refinement pipeline:

- Generates initial synthetic data by independently sampling each column from its empirical marginal distribution
- Trains a Random Forest discriminator to classify real vs synthetic rows
- Refines synthetic data using **leaf-conditional sampling** from the fitted forest
- Repeats until the discriminator can no longer distinguish real from synthetic data (OOB accuracy ≤ 0.5 + δ)

### Key Idea
At convergence, the discriminator's leaf structure encodes the joint dependency between features. Synthetic points are refined by finding real training rows that share the same leaf assignments across trees, then sampling feature values from those neighbours independently per feature. This captures complex multivariate structure without parametric assumptions.

---

## Leaf Matching Strategy
The implementation uses **majority-vote leaf matching** rather than requiring all-tree agreement:

- For each synthetic point, counts how many trees place each real row in the same leaf
- Accepts any real row that agrees on ≥ `leaf_thresh` fraction of trees (default 0.5)
- Falls back to the top 10% most similar rows if no match is found

This makes the method robust on larger and higher-dimensional datasets where strict all-tree matching rarely finds candidates.

---

## Training Details
- Adversarial loop using `sklearn.ensemble.RandomForestClassifier`
- OOB accuracy used as the convergence signal (no held-out set required)
- A single persistent RNG is advanced across all iterations so each round produces genuinely different synthetic data

### Convergence Criteria
Training stops when either:
- OOB accuracy ≤ 0.5 + `delta` (forest no longer distinguishes real from synthetic)
- OOB accuracy stops improving between rounds (early stopping)
- `max_iters` is reached

---

## Hyperparameters
Defined in `ARFModel`:

- `num_trees = 30`
- `max_iters = 10`
- `delta = 0.0`
- `min_node_size = 5`
- `leaf_thresh = 0.5`
- `verbose = True`
- `seed = 42`

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

Or alternatively a single `train_full.csv` with the label as the last column.

---

## Preprocessing
The internal `_ARFEngine` automatically handles mixed tabular data types:

### Numerical Features
- Passed through as-is (float cast)

### Categorical Features
- Integer-coded using `sklearn.preprocessing.LabelEncoder`
- Decoded back to original string labels after generation

---

## Label Handling
Labels are **not** passed through the ARF generative process. Instead:
- `y` is stored from training
- Synthetic labels are sampled from the empirical training label distribution using `sample(replace=True)`

This guarantees all classes appear in correct proportions regardless of rarity.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Evaluation
`evaluate()` returns the **mean column-wise KS statistic** between real and synthetic numeric features. Lower is better; 0 indicates identical marginal distributions.

---

## Strengths
- Non-parametric — no distributional assumptions
- Captures joint feature dependencies via leaf structure
- Fully self-contained (numpy, pandas, scikit-learn only)
- Fast convergence on low-dimensional datasets

---

## Limitations
⚠️ The model may struggle with:
- Very large datasets (leaf-matching loop is O(n_synth × n_real) per iteration)
- Datasets where the discriminator never converges (hits `max_iters`)
- Purely continuous high-dimensional data with complex dependencies

---

## Usage

```python
from models import ARFModel

model = ARFModel(num_trees=30, max_iters=10)

model.train(
    data_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```
