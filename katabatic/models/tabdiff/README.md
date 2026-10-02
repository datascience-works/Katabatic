# TabDiff

TabDiff is a diffusion model for mixed-type tabular data that diffuses numeric and categorical columns **jointly** in a single process, rather than running one process per column type.

***

## Overview

Katabatic already ships TabDDPM, which runs two separate diffusion processes: Gaussian for numeric columns and multinomial for categorical columns. TabDiff's distinguishing idea is a single, unified diffusion over the whole joint feature space. This implementation, **adapted for Katabatic**:

- **Joint encoding**: numeric columns are z-scored, categorical columns are one-hot encoded, and everything is concatenated into one continuous vector.
- **Single Gaussian diffusion**: one class-conditional DDPM noises and denoises that joint vector with a cosine schedule.
- **Decode step**: categorical blocks are recovered with `argmax`; numeric columns are un-scaled, clipped to the training range and rounded if they were integer-valued in training.
- **Self-contained**: trains from scratch, with no pretrained weights and no dependency outside the Katabatic framework (only `torch` and `scikit-learn`).

***

## Implementation Details

**Paper**: Shi, J., Xu, M., Hua, H., Zhang, H., Ermon, S. and Leskovec, J. (2024). *TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation.* ICLR 2025. https://arxiv.org/abs/2410.20626

### Status: simplified implementation

This is a working but **simplified** version of TabDiff, not a full reproduction of the paper:

- The paper learns **per-feature-type adaptive noise schedules** and uses **continuous-time** diffusion. This implementation uses a **discrete-time** DDPM with a single shared cosine schedule across the whole joint vector.
- The paper's denoiser is a transformer over per-column tokens. This uses a small MLP over the flattened joint vector, with less per-column inductive bias.
- Categorical columns are recovered with hard `argmax` instead of the paper's learned discretisation.

The mechanism that distinguishes TabDiff from TabDDPM (one diffusion process over the joint space instead of two) is implemented. Results below should be read as those of this simplified model, not of the published one.

### Katabatic model structure

```
katabatic/models/tabdiff/
  __init__.py   exposes Tabdiff
  models.py     training loop and reverse sampling
  utils.py      JointEncoder, DenoiserMLP, cosine noise schedule
```

### Hyperparameter comparison

| Parameter | Class default (`Tabdiff._defaults`) | **Benchmark scripts** |
|---|---|---|
| steps | 300 | **3000** |
| num_timesteps | 100 | **100** |
| hidden | 128 | **256** |
| batch_size | 128 | **256** |
| lr | 1e-3 | **1e-3** |
| seed | 42 | **42** |
| categorical_cols | inferred from dtype / `info.json` | **passed explicitly from the dataset metadata** |

Passing `categorical_cols` matters: the benchmark runner hands every model integer-coded columns, so inferring roles from dtype would treat categorical codes as continuous.

***

## Installation

```bash
poetry install --extras tabdiff
```

***

## Usage

Benchmark scripts for each dataset (also proposed for `development` in pull request [#268](https://github.com/datascience-works/Katabatic/pull/268), where they live under `benchmarks/examples/tabdiff/`):
- Adult [benchmarks/examples/tabdiff/run_tabdiff_adult.py](../../../benchmarks/tabdiff/run_tabdiff_adult.py)
- Shuttle [benchmarks/examples/tabdiff/run_tabdiff_shuttle.py](../../../benchmarks/tabdiff/run_tabdiff_shuttle.py)
- Car [benchmarks/examples/tabdiff/run_tabdiff_car.py](../../../benchmarks/tabdiff/run_tabdiff_car.py)
- Magic [benchmarks/examples/tabdiff/run_tabdiff_magic.py](../../../benchmarks/tabdiff/run_tabdiff_magic.py)
- Nursery [benchmarks/examples/tabdiff/run_tabdiff_nursery.py](../../../benchmarks/tabdiff/run_tabdiff_nursery.py)

```python
from katabatic.models.tabdiff import Tabdiff

model = Tabdiff(config={"steps": 3000, "hidden": 256, "batch_size": 256,
                        "categorical_cols": ["workclass", "education"]})  # optional

model.train("path/to/split_dir", synthetic_dir="path/to/synthetic_dir")
synthetic_df = model.sample(len(real_df))
```

`train()` reads `x_train.csv` and `y_train.csv` from the directory and writes `x_synth.csv` / `y_synth.csv`.

***

## Model Evaluation Benchmark Results

Evaluated with the Katabatic evaluation pipeline on all five required datasets, six dimensions each (seed 42).

| Dataset | Composite | Fidelity | Utility | Diversity | Privacy | Consistency | Stability |
|---|---|---|---|---|---|---|---|
| Car | 0.8630 | 0.9446 | 0.9161 | 0.9925 | 0.4617 | 0.8890 | 0.9765 |
| Magic | 0.8788 | 0.8982 | 0.9628 | 0.8131 | 0.7820 | 0.6913 | 0.9915 |
| Nursery | 0.8472 | 0.9149 | 0.9030 | 0.9876 | 0.4677 | 0.8457 | 0.9780 |
| Shuttle | 0.9195 | 0.9748 | 0.9932 | 0.9501 | 0.8661 | 0.5374 | 0.9905 |
| Adult | 0.7955 | 0.7337 | 0.8406 | 0.8701 | 0.9848 | 0.3335 | 0.9960 |

### Note on the numeric decode step

The first Magic run scored a composite of **0.7303** (fidelity 0.5128, diversity 0.5000, consistency 0.2455). The runner gives models integer-binned columns (0 to 12), but the unbounded sampler produced floats with extreme outliers (for example `fLength` up to 519 against a real maximum of 9). Clipping decoded numeric columns to the training range and rounding integer-valued columns raised Magic to **0.8788**. The table above uses the final code for all five datasets.

***

## Model Performance

> **Hardware & runtime:** CPU only (the model does not use the Apple MPS GPU). Runtime includes preprocessing, training, sampling and all six evaluation dimensions.

| Dataset | Runtime (s) |
|---|---|
| Car | 13.9 |
| Nursery | 50.3 |
| Magic | 105.8 |
| Shuttle | 113.5 |
| Adult | 267.0 |

Hardware: macOS (Darwin 25.6.0), Apple Silicon (arm64), 17.2 GB RAM, no CUDA GPU.

Output files (written to `benchmarks/results/<dataset>/tabdiff/`):
- `tabdiff_<dataset>_evaluation_report.json`
- `tabdiff_<dataset>_evaluation_summary.csv`

***

## Strengths
- Handles numeric and categorical columns in one process, with no separate per-type models.
- Trains from scratch in minutes on a laptop CPU.
- Strong utility, diversity and stability on most datasets.

## Limitations
- Simplified relative to the paper (discrete-time, shared noise schedule, MLP denoiser); see Status above.
- **Consistency is the weakest dimension** (0.33 on Adult, 0.54 on Shuttle): relationships between columns are only partly preserved.
- **Privacy is low on the small, fully categorical datasets** (about 0.46 on Car and Nursery). We did not investigate the cause; small, low-cardinality datasets make exact record overlap more likely.
- One-hot encoding grows the input with the number of categories, so very high-cardinality columns are costly.
- Results are from a single seed and a fixed step budget; they were not tuned per dataset.
