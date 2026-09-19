# TabEBM Model

## Model Overview

TabEBM (Tabular Energy-Based Model) is a method for generating synthetic tabular data. It creates **one separate model for each class label** in the dataset, which helps it learn the unique patterns of each class independently. It was published at NeurIPS 2024 by researchers at the University of Cambridge.

Instead of learning a shared model for all classes, TabEBM trains a small binary classifier per class, then uses a sampling process called SGLD to generate new data points that look like they belong to that class.

---

### Key Idea

For each class, TabEBM:
1. Takes the real samples from that class as **positive examples**
2. Creates fake "negative" samples placed far away from the real data (at the corners of a hypercube)
3. Trains a binary classifier (TabPFN) to tell real vs. fake apart — this is the **surrogate task**
4. Uses the trained classifier's output as an **energy function**: low energy = looks like real data
5. Runs **SGLD sampling** to generate new points that move toward low-energy (realistic) regions

The SGLD update at each step is:

`x_new = x_old - (step_size / 2) × gradient_of_energy + small_random_noise`

This gradually pushes a starting point toward realistic-looking data while adding a little noise to keep diversity.

---

### Research Paper

**Title:** TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models  
**Authors:** Andrei Margeloiu, Xiangjian Jiang, Nikola Simidjievski, Mateja Jamnik  
**Venue:** NeurIPS 2024  
**Link:** https://arxiv.org/abs/2409.16118

**Parameters we kept from the paper:**
- `sgld_steps = 200` — number of SGLD iterations (paper default)
- `sgld_step_size = 0.01` — gradient step size (paper default)
- `sgld_noise_std = 0.01` — noise at each SGLD step (paper default)
- `distance_negative_class = 5.0` — how far to place negative samples (paper default)
- `starting_point_noise_std = 0.01` — noise on starting points (paper default)

**What we changed for benchmarking:**
- `max_data_size = 1000` — reduced from 10000 for faster testing
- For large datasets (Credit Card, Covertype), we subsampled 2000 rows per class to avoid memory issues

---

## Approach

### Training Details
- Data is normalised before training (continuous columns: z-score, categorical columns: ordinal encoding)
- For each class, a binary TabPFN classifier is trained on real samples (label=1) vs. hypercube negative samples (label=0)
- TabPFN is a pre-trained transformer — it does not need gradient-based training itself, which makes this fast
- The energy function is derived directly from the classifier's output probabilities

### Convergence Criteria
- Training ends after TabPFN is fit (it is a zero-shot/in-context model, no iterative training loop)
- SGLD sampling runs for a fixed number of steps (`sgld_steps = 200`)

---

## Hyperparameters

Defined in `TabEBMConfig` in `models.py`:

| Parameter | Default | Description |
|---|---|---|
| `max_data_size` | 10000 | Max training samples per class (we use 1000 for testing) |
| `starting_point_noise_std` | 0.01 | Noise added to starting points before SGLD |
| `sgld_step_size` | 0.01 | Step size for each SGLD gradient update |
| `sgld_noise_std` | 0.01 | Noise injected at each SGLD step |
| `sgld_steps` | 200 | Number of SGLD iterations per sample |
| `distance_negative_class` | 5.0 | Distance (in std devs) to place hypercube negative samples |
| `seed` | 42 | Random seed |

---

## Input

- `X`: Feature matrix (tabular, numeric)
- `y`: Class labels

### Expected Files
- `x_train.csv` — features only, no label column
- `y_train.csv` — label column only

> Note: TabEBM requires the features and labels in **two separate CSV files**, unlike some other models that use a single combined file.

---

## Preprocessing

### Numerical Features
Continuous features are z-score normalised (mean=0, std=1) before training.

### Categorical Features
Categorical features are ordinal encoded (each category gets an integer value). TabEBM treats all input as numeric internally.

---

## Label Handling

Labels are used to split the training data by class. TabEBM trains one separate model for each unique class value. Labels are passed via `y_train.csv`.

---

## Output

Generated files saved to `synthetic_dir`:

- `x_synth.csv` — synthetic feature rows
- `y_synth.csv` — synthetic labels
- `metadata.json` — schema and column info

---

## Strengths

- **Class-specific models** mean each class gets its own generator — useful when classes have very different distributions
- **No neural network training from scratch** — uses pre-trained TabPFN, so it is relatively fast
- **Works well with small datasets** — TabPFN is designed for small tabular data
- **Good fidelity scores** across all datasets tested

---

## Limitations

- **Stability = 0.0** in all our benchmarks: because SGLD is a stochastic (random) process, results change each run. This makes it hard to reproduce exact outputs.
- **Not suitable for very large datasets** without subsampling — the in-context learning in TabPFN has memory limits
- **Privacy scores are lower** on some datasets (e.g. Adult, Bank Marketing) because the class-specific models can memorise small classes
- **Consistency scores vary** — some feature correlations may not be perfectly preserved

---

## Installation

```bash
poetry install --extras tabebm
```

---

## Usage

```python
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig

config = TabEBMConfig(
    max_data_size=1000,
    starting_point_noise_std=0.01,
    sgld_step_size=0.01,
    sgld_noise_std=0.01,
    sgld_steps=200,
    distance_negative_class=5.0,
    seed=42,
)

model = TabEBMModel(target_col="label", config=config)

model.train(
    output_dir="path/to/split_dir",    # folder with x_train.csv and y_train.csv
    synthetic_dir="path/to/synth_dir"
)

x_synth, y_synth = model.sample(1000)
```

**Benchmark scripts for each dataset:**
- Adult: [`benchmarks/examples/tabebm/run_tabebm_adult.py`](benchmarks/examples/tabebm/run_tabebm_adult.py)
- Bank Marketing: [`benchmarks/examples/tabebm/run_tabebm_bank_marketing.py`](benchmarks/examples/tabebm/run_tabebm_bank_marketing.py)
- Car: [`benchmarks/examples/tabebm/run_tabebm_car.py`](benchmarks/examples/tabebm/run_tabebm_car.py)
- Credit Card: [`benchmarks/examples/tabebm/run_tabebm_creditcard.py`](benchmarks/examples/tabebm/run_tabebm_creditcard.py)
- Covertype: [`benchmarks/examples/tabebm/run_tabebm_covtype.py`](benchmarks/examples/tabebm/run_tabebm_covtype.py)

---

## Model Evaluation Benchmark Results

> Note: Stability = 0.0 for all datasets. This is expected — TabEBM uses stochastic SGLD sampling, so results differ slightly each run.

#### Adult Dataset

Composite score: **0.7477**

| Dimension | Score |
|---|---|
| Fidelity | 0.9482 |
| Utility | 0.9304 |
| Diversity | 0.8208 |
| Privacy | 0.4299 |
| Consistency | 0.3844 |
| Stability | 0.0000 |

---

#### Bank Marketing Dataset

Composite score: **0.7757**

| Dimension | Score |
|---|---|
| Fidelity | 0.9823 |
| Utility | 0.9683 |
| Diversity | 0.8583 |
| Privacy | 0.4291 |
| Consistency | 0.4098 |
| Stability | 0.0000 |

---

#### Car Dataset

Composite score: **0.8349**

| Dimension | Score |
|---|---|
| Fidelity | 0.9836 |
| Utility | 0.9785 |
| Diversity | 0.9993 |
| Privacy | 0.3333 |
| Consistency | 0.9664 |
| Stability | 0.0000 |

---

#### Credit Card Dataset

Composite score: **0.8623**

| Dimension | Score |
|---|---|
| Fidelity | 0.9311 |
| Utility | 1.0000 |
| Diversity | 0.8791 |
| Privacy | 0.9059 |
| Consistency | 0.5573 |
| Stability | 0.0000 |

---

#### Covertype Dataset

Composite score: **0.7603**

| Dimension | Score |
|---|---|
| Fidelity | 0.9765 |
| Utility | 0.8061 |
| Diversity | 0.6716 |
| Privacy | 0.6434 |
| Consistency | 0.7042 |
| Stability | 0.0000 |

---

## Model Performance

- Runs on CPU (no GPU required)
- Adult (48K rows, 15 cols): ~5–10 minutes
- Covertype (464K rows, 54 cols, subsampled to 14K): ~10–15 minutes
- Credit Card (284K rows, 31 cols, subsampled to 4K): ~5–10 minutes
