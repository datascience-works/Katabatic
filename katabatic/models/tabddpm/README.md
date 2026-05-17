# TabDDPM Model

## Model Overview
TabDDPM is a **diffusion-based deep generative model** for tabular data that applies a Gaussian denoising diffusion process to learn and reproduce the joint distribution of mixed numerical and categorical features.

The model uses a **Variance Preserving SDE** forward process and an **MLP denoiser** conditioned on class labels, trained with a mixed reconstruction loss across numerical and categorical feature dimensions.

This implementation prefers the external `tabddpm` package if installed, and falls back to a lightweight local implementation in `utils.py` otherwise.

---

## Approach
The model follows a standard DDPM pipeline adapted for tabular data:

- Applies a forward VP-SDE noise schedule to corrupt training data across `num_timesteps` steps
- Trains an MLP denoiser conditioned on class labels to predict the clean signal at each noise level
- Generates synthetic data by running the **reverse SDE** (Euler-Maruyama) from Gaussian noise to clean samples
- Optionally maintains an **EMA (Exponential Moving Average)** copy of the denoiser for improved sample quality

### Forward Process (VP-SDE)

```math
dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW
```

### Mixed Loss
Training minimises a combined loss over numerical and categorical dimensions:

```math
\mathcal{L} = \mathcal{L}_{gauss} + \mathcal{L}_{multi}
```

- **Gaussian loss** (MSE): on numerical feature dimensions
- **Multinomial loss** (L1): on categorical feature dimensions

---

## Architecture
The denoiser is a **class-conditioned MLP**:

- Input: noisy features concatenated with one-hot class label embedding
- Hidden layers: configurable depth and width via `d_layers`
- Output: denoised feature vector of the same dimension as input

---

## Hyperparameters
Defined in `Tabddpm._defaults`:

- `steps = 10000`
- `lr = 1e-3`
- `weight_decay = 1e-5`
- `batch_size = 256`
- `num_timesteps = 1000`
- `gaussian_loss_type = "mse"`
- `scheduler = "cosine"`
- `d_layers = (256, 256, 256, 256)`
- `dropout = 0.0`
- `seed = 42`
- `use_ema = True`

A lighter default config is used in pipeline mode for stability:

- `steps = 200`
- `num_timesteps = 100`
- `batch_size = 32`
- `d_layers = (64, 64)`
- `use_ema = False`

---

## Input
- `X`: Tabular feature matrix (DataFrame or ndarray)
- `y`: Target labels (Series or ndarray)

### Expected Files
- `x_train.csv`
- `y_train.csv`

---

## Preprocessing
The model automatically handles mixed tabular data types:

### Numerical Features
- Passed through as float32

### Categorical Features
- Detected from object/category dtype columns
- Integer-coded using `sklearn.preprocessing.LabelEncoder` per column
- Decoded back to original string labels after generation

### Label Encoding
- Classification labels are integer-encoded via `LabelEncoder` for conditioning
- Task type (classification vs regression) is inferred automatically

---

## Label Handling
Labels are **not** generated through the diffusion process. Instead:
- Encoded training labels are stored during `train()`
- Synthetic labels are sampled empirically from the training class distribution

This guarantees all classes appear in correct proportions regardless of rarity, and avoids class boundary artefacts from continuous diffusion outputs.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`

---

## Evaluation
`evaluate()` returns a **negative diffusion loss** averaged over a configurable number of mini-batches. Higher is better (less negative = lower loss = better fit).

---

## Strengths
- Principled probabilistic generative model
- Class-conditional generation with learned label conditioning
- EMA weight averaging improves sample quality
- Handles both numerical and categorical features natively

---

## Limitations
⚠️ The model may struggle with:
- Very rare classes (low representation in training batches)
- High-cardinality categorical features (each category is a separate integer code)
- Small datasets where the MLP overfits before convergence

⚠️ Training is computationally expensive relative to non-deep models. The pipeline default config uses a reduced step count for practical runtimes.

---

## Usage

```python
from models import Tabddpm

model = Tabddpm()

model.train(
    "path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```
