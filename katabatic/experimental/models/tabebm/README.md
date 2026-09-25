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
- **High fidelity** in benchmarks (0.96–0.99), largely because samples stay on or next to real training rows.

---

## Limitations

- **SGLD sampling cost grows with dataset size**: every step computes distances between each synthetic point and each (subsampled) training point of its class, so runtime scales with *synthetic rows × `max_data_size` × features × `sgld_steps`*. Memory is bounded (the gradient is computed in chunks), but large datasets such as Adult and Shuttle still take a long time on CPU. Lower `max_data_size` or `sgld_steps` to trade quality for speed.
- **Samples reproduce training rows.** The energy gradient pulls each sample towards its nearest training row. On all-categorical data (car, nursery, and magic as packaged) rounding then snaps it exactly onto that row, so 100% of synthetic rows are exact copies. On mixed data (shuttle, adult) 100% are near-duplicates and a classifier separates synthetic from real rows almost perfectly. Do not use this model where training records must not be disclosed.
- **Consistency scores vary** — some feature correlations may not be perfectly preserved
- **`sample()` always reuses the configured `seed`** (`TabEBMConfig.seed`, default 42) for every call, so repeated calls on the same fitted model are fully deterministic — this makes results reproducible run-to-run, but also means the stability evaluation dimension (which expects independent runs) isn't measuring genuine run-to-run variance for this model

---

## Status

TabEBM is experimental (`supported: False` in `ModelRegistry`). It passes the promotion contract, but its samples reproduce training rows (see Limitations), so it stays experimental until the sampler generates new rows.

```bash
pip install katabatic[tabebm]   # or: poetry install -E tabebm
```

---

## Usage

```python
from katabatic.experimental.models.tabebm.models import TabEBMModel, TabEBMConfig

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
- Adult: [`benchmarks/examples/tabebm/run_tabebm_adult.py`](../../../../benchmarks/examples/tabebm/run_tabebm_adult.py)
- Car: [`benchmarks/examples/tabebm/run_tabebm_car.py`](../../../../benchmarks/examples/tabebm/run_tabebm_car.py)
- Magic: [`benchmarks/examples/tabebm/run_tabebm_magic.py`](../../../../benchmarks/examples/tabebm/run_tabebm_magic.py)
- Nursery: [`benchmarks/examples/tabebm/run_tabebm_nursery.py`](../../../../benchmarks/examples/tabebm/run_tabebm_nursery.py)
- Shuttle: [`benchmarks/examples/tabebm/run_tabebm_shuttle.py`](../../../../benchmarks/examples/tabebm/run_tabebm_shuttle.py)

---

## Evaluation

`model.evaluate(real_df, target_col=..., test_data=...)`, inherited from `Model`, scores the fitted model on Katabatic's six dimensions (fidelity, utility via TSTR, diversity, privacy, consistency and stability) and returns an `EvaluationReport` (`report.dimension_scores`, `report.composite_score`). Stability is always 1.0 for TabEBM because `sample()` reuses the configured seed (see Limitations).

---

## Model Evaluation Benchmark Results

| Dataset | Fidelity | Utility | Diversity | Privacy | Consistency | Stability | Composite | Runtime |
|---|---|---|---|---|---|---|---|---|
| Car | 0.9838 | 0.9969 | 0.9989 | 0.3333 | 0.9406 | 1.0000 | **0.8888** | 3 min |
| Nursery | 0.9844 | 0.9623 | 0.9986 | 0.3333 | 0.4983 | 1.0000 | **0.8326** | 15 min |
| Magic | 0.9887 | 0.9054 | 0.8683 | 0.3333 | 0.4723 | 1.0000 | **0.7981** | 42 min |
| Shuttle | 0.9887 | 0.7677 | 0.9466 | 0.6413 | 0.4918 | 1.0000 | **0.8059** | 1 h 59 min |
| Adult | 0.9570 | 0.9188 | 0.8658 | 0.3853 | 0.3935 | 1.0000 | **0.7946** | 1 h 33 min |

How to read these scores:

| Dataset | Exact copies of training rows | Near-duplicates (Gower < 0.01) | Real-vs-synthetic classifier accuracy | TSTR / TRTR accuracy |
|---|---|---|---|---|
| Car | 100% | 0% | 0.44 | 0.856 / 0.861 |
| Nursery | 100% | 0% | 0.94 | 0.859 / 0.896 |
| Magic | 100% | 0% | 1.00 | 0.712 / 0.827 |
| Shuttle | 0% | 100% | 1.00 | 0.740 / 0.978 |
| Adult | 0% | 100% | 1.00 | 0.770 / 0.836 |

- **Fidelity is high because the samples are (near-)copies of the training data**, not because the model has learned the distribution. The same explains TSTR matching TRTR on car. Privacy is correspondingly low.
- **Stability is always 1.0** because `sample()` reuses the configured seed, so the stability runs are identical (see Limitations). It says nothing about run-to-run variance.
- **Classifier accuracy on nursery and magic is high even though every row is a real training row.** So the model doesn't reproduce training rows in their real proportions.
