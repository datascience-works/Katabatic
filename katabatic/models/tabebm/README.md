# TabEBM Model

## Model Overview

TabEBM (Tabular Energy-Based Model) is a method for generating synthetic tabular data. It creates **one separate model for each class label** in the dataset, which helps it learn the unique patterns of each class independently. It was published at NeurIPS 2024 by researchers at the University of Cambridge.

> **Implementation note:** the paper's method trains a binary TabPFN classifier per class and uses its output as the energy function. The Katabatic implementation (`_TabEBMBackend` in `models.py`) does **not** use TabPFN — it's a from-scratch NumPy energy approximation instead, explicitly to avoid TabPFN's torch/autograd dependency. The energy gradient at each point is approximated directly from distances to the real class samples and to synthetic hypercube negatives (see Key Idea below), with no classifier trained at all. This is a substantially different (and less expensive) approximation of the paper's method, not TabPFN under a different name — keep that in mind when comparing results to the paper.

---

### Key Idea

For each class, TabEBM:
1. Takes the real samples from that class as **positive examples**
2. Creates 4 fake "negative" samples placed at random corners of a hypercube, `distance_negative_class` standard deviations out
3. Approximates an energy gradient at each candidate point directly from distances — the gradient toward the single nearest real/negative point, plus the mean gradient toward all real positive points — with no classifier in the loop
4. Runs **SGLD sampling**: repeatedly nudges points against that gradient (toward realistic regions) with added noise, for a fixed number of steps

The SGLD update at each step is:

`x_new = x_old - step_size × gradient_estimate + small_random_noise`

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

---

## Approach

### Training Details
- Data is normalised before training (continuous columns: z-score, categorical columns: ordinal encoding)
- No classifier is trained. For each class, the energy gradient used by SGLD is computed directly from pairwise distances between candidate points and that class's real + surrogate-negative samples (see the implementation note above)
- "Training" is therefore just fitting the schema/encoding and caching the (encoded) training data for use at sampling time — there's no optimisation loop

### Convergence Criteria
- There is no training loop to converge sd 0-`train()` completes as soon as the schema is fit
- SGLD sampling runs for a fixed number of steps (`sgld_steps = 200`)

---

## Hyperparameters

Defined in `TabEBMConfig` in `models.py`:

| Parameter | Default | Description |
|---|---|---|
| `max_data_size` | 10000 | Max training rows used for sampling; larger training sets are stratified-subsampled down to this size (every class is kept, however rare). The benchmark scripts use 1000 |
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
- **No classifier or neural network training at all** — sampling only needs the cached training data, so `train()` itself is nearly instant
- **Good fidelity scores** in benchmarks so far

---

## Limitations

- **SGLD sampling cost grows with dataset size**: every step computes distances between each synthetic point and each (subsampled) training point of its class, so runtime scales with *synthetic rows × `max_data_size` × features × `sgld_steps`*. Memory is bounded (the gradient is computed in chunks), but large datasets such as Adult and Shuttle still take a long time on CPU. Lower `max_data_size` or `sgld_steps` to trade quality for speed.
- **Privacy scores are lower** on some datasets (e.g. Adult) because the class-specific generation can memorise small classes
- **Consistency scores vary** — some feature correlations may not be perfectly preserved
- **`sample()` always reuses the configured `seed`** (`TabEBMConfig.seed`, default 42) for every call, so repeated calls on the same fitted model are fully deterministic — this makes results reproducible run-to-run, but also means the stability evaluation dimension (which expects independent runs) isn't measuring genuine run-to-run variance for this model

---

## Status

TabEBM is an officially supported Katabatic model (`supported: True` in `ModelRegistry`).

```bash
pip install katabatic[tabebm]   # or: poetry install -E tabebm
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
    "path/to/split_dir",  # folder with x_train.csv and y_train.csv
    synthetic_dir="path/to/synth_dir",
)

synthetic_df = model.sample(1000)  # single DataFrame: features + target as the last column
```

Through the artifact pipeline (recommended — also persists state for `load_from_ref`):

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

TrainTestSplitPipeline(model=TabEBMModel()).run(
    input_csv="data.csv",
    dataset_name="mydata",
    artifact_store=store,
    model_name="tabebm",
)
```

**Benchmark scripts:**
- Adult: [`benchmarks/examples/tabebm/run_tabebm_adult.py`](benchmarks/examples/tabebm/run_tabebm_adult.py)
- Car: [`benchmarks/examples/tabebm/run_tabebm_car.py`](benchmarks/examples/tabebm/run_tabebm_car.py)
- Magic: [`benchmarks/examples/tabebm/run_tabebm_magic.py`](benchmarks/examples/tabebm/run_tabebm_magic.py)
- Nursery: [`benchmarks/examples/tabebm/run_tabebm_nursery.py`](benchmarks/examples/tabebm/run_tabebm_nursery.py)
- Shuttle: [`benchmarks/examples/tabebm/run_tabebm_shuttle.py`](benchmarks/examples/tabebm/run_tabebm_shuttle.py)

---

## Model Evaluation Benchmark Results


#### Car Dataset

Composite score: **0.8888**

| Dimension | Score |
|---|---|
| Fidelity | 0.9838 |
| Utility | 0.9969 |
| Diversity | 0.9989 |
| Privacy | 0.3333 |
| Consistency | 0.9406 |
| Stability | 1.0000 |

---

#### Adult Dataset

Not yet re-verified after the promotion fixes (long-running; see Model Performance below). Whoever picks up the next Validation & Benchmarking pass should fill this in from a completed `run_tabebm_adult.py` run.

---
