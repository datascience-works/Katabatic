# SMOTE Model

## Model Overview
SMOTE (Synthetic Minority Over-sampling Technique) is a **classical oversampling method** for tabular data that generates new minority-class samples by interpolating between existing minority-class observations and their nearest neighbours.

This implementation is **fully self-contained** using only numpy, pandas, and scikit-learn — no `imbalanced-learn` dependency is required.

---

## Approach
The model follows the original Chawla et al. (2002) algorithm:

- For each minority-class sample, find its k nearest neighbours within the same class
- Pick one neighbour at random
- Generate a new synthetic point by linear interpolation between the sample and its chosen neighbour

### Key Idea
Rather than simply duplicating minority samples, SMOTE creates new points in the feature space *between* existing minority samples. This encourages classifiers to learn a broader, more generalised decision boundary for the minority class.

### Interpolation Formula

```math
x_{new} = x_i + \lambda \cdot (x_{neighbour} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)
```

---

## Sampling Strategy
The `sampling_strategy` parameter controls how many synthetic samples are generated per class:

- `"auto"` (default): oversample all minority classes up to the majority class count
- `float`: sets the desired minority/majority ratio

After oversampling, the synthetic set is subsampled back to the original training size to ensure fair comparison with other Katabatic models.

---

## Hyperparameters
Defined in `SMOTEModel`:

- `k_neighbors = 5`
- `sampling_strategy = "auto"`
- `random_state = 42`

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
SMOTE operates directly on the encoded feature matrix. The pipeline automatically handles:

### Minority Class Detection
- Counts per-class samples to identify all classes below the majority class count
- Automatically reduces `k_neighbors` if a minority class has fewer samples than requested k

### Nearest Neighbour Search
- Uses `sklearn.neighbors.NearestNeighbors` with auto algorithm selection

---

## Label Handling
SMOTE generates synthetic labels to match the interpolated feature rows — new samples for class `c` are labelled `c`. Labels are not sampled independently; they are determined by which minority class is being oversampled.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

The metadata records the adjusted k, sampling strategy, original training size, number of new rows generated, and number of rows returned.

---

## Evaluation
`evaluate()` returns the **mean column-wise KS statistic** between real and synthetic numeric features. Lower is better; 0 indicates identical marginal distributions.

---

## Strengths
- Simple, fast, and interpretable
- No training phase — oversampling happens in a single pass
- Fully self-contained (no imbalanced-learn required)
- Robust handling of small minority classes via automatic k adjustment

---

## Limitations
⚠️ The model may struggle with:
- Highly imbalanced datasets with very few minority samples (k may be forced to 1)
- Datasets with mixed continuous and categorical features (interpolation is applied uniformly)
- Datasets where minority class boundaries are not convex

⚠️ SMOTE does not model the full joint distribution — it only oversamples minority classes and cannot generate majority-class synthetic data.

---

## Usage

```python
from models import SMOTEModel

model = SMOTEModel(k_neighbors=5)

model.train(
    output_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```
