# TabEBM Model

## Model Overview
TabEBM generates synthetic tabular data using Energy-Based Models (EBMs). It trains one EBM per class and uses Stochastic Gradient Langevin Dynamics (SGLD) to generate new samples that match the distribution of the real data.

---

### Key Idea
The model assigns an "energy" score to each data point — low energy means the point looks real, high energy means it looks fake. It learns this by training on real data (low energy) vs generated data (high energy). New synthetic samples are created by starting from a real data point and gradually moving it toward lower-energy regions using SGLD.

---

### Research Paper
**TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models** (2024)

Parameters kept the same as the paper:
- `sgld_step_size` = 0.01
- `sgld_noise_std` = 0.01
- `sgld_steps` = 200
- `starting_point_noise_std` = 0.01
- `distance_negative_class` = 5.0

Parts not fully specified in the paper that had to be inferred:
- Gradient clipping during SGLD (we used `[-5, 5]`) to prevent numerical instability.

---

## Approach
1. Normalise continuous features and encode categorical features.
2. Train one EBM per class using real data as positive examples and out-of-class data as negative examples.
3. Generate synthetic samples by running SGLD chains starting from real data points.
4. Decode the output back to original feature values.

### Training Details
For each class, the model trains a small neural network (energy function) using contrastive divergence — pushing real samples to low energy and generated samples to high energy.

### Convergence Criteria
Training runs for a fixed 200 SGLD steps as specified in the paper. There is no early stopping.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `sgld_step_size` | `0.01` | SGLD step size |
| `sgld_noise_std` | `0.01` | Noise added at each SGLD step |
| `sgld_steps` | `200` | Number of SGLD steps |
| `starting_point_noise_std` | `0.01` | Noise added to starting points |
| `distance_negative_class` | `5.0` | Scaling for negative class chains |
| `max_data_size` | `None` | Max rows per class (None = use all) |
| `seed` | `42` | Random seed |

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

---

## Preprocessing

### Numerical Features
Standardised using a standard scaler fitted on training data.

### Categorical Features
Ordinal-encoded to integers before training, then decoded back to original labels after generation.

> **Bug fix:** pandas 2.x changed the dtype name for string columns from `object` to `str`. The original code did not detect this, causing categorical columns to appear as NaN in the output. This has been fixed in `utils.py` and `models.py`.

---

## Label Handling
The model trains one EBM per class. Labels are read from `y_train.csv` and used to split the data by class. During sampling, each class generates samples separately and they are combined.

---

## Output
- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Evaluation
`evaluate()` returns a report with a composite score and individual scores for fidelity, utility, diversity, privacy, consistency, and stability.

---

## Strengths
- Trains a separate model per class, so class-specific patterns are preserved.
- Works with both categorical-only and continuous-only datasets.
- Consistently high fidelity scores.

---

## Installation

```bash
poetry install --extras tabebm
```

---

## Usage

Benchmark scripts:
- Adult: [benchmarks/examples/tabebm/run_tabebm_adult.py](benchmarks/examples/tabebm/run_tabebm_adult.py)
- Bank Marketing: [benchmarks/examples/tabebm/run_tabebm_bank_marketing.py](benchmarks/examples/tabebm/run_tabebm_bank_marketing.py)
- Car: [benchmarks/examples/tabebm/run_tabebm_car.py](benchmarks/examples/tabebm/run_tabebm_car.py)
- Credit Card: [benchmarks/examples/tabebm/run_tabebm_creditcard.py](benchmarks/examples/tabebm/run_tabebm_creditcard.py)
- Covertype: [benchmarks/examples/tabebm/run_tabebm_covtype.py](benchmarks/examples/tabebm/run_tabebm_covtype.py)

```python
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig

config = TabEBMConfig(
    sgld_step_size=0.01,
    sgld_noise_std=0.01,
    sgld_steps=200,
    starting_point_noise_std=0.01,
    distance_negative_class=5.0,
    seed=42,
)

model = TabEBMModel(target_col=target_col, config=config)

model.train(
    output_dir="path_to_split_dir",
    synthetic_dir="path_to_save"
)

x_synth, y_synth = model.sample(1000)
```

---

## Model Evaluation Benchmarks Results

#### Adult Dataset
Composite score: 0.7477
- fidelity      0.9482
- utility       0.9304
- diversity     0.8208
- privacy       0.4299
- consistency   0.3844
- stability     0.0000

#### Car Dataset
Composite score: 0.8349
- fidelity      0.9836
- utility       0.9785
- diversity     0.9993
- privacy       0.3333
- consistency   0.9664
- stability     0.0000

#### Credit Card Dataset
Composite score: 0.8623
- fidelity      0.9311
- utility       1.0000
- diversity     0.8791
- privacy       0.9059
- consistency   0.5573
- stability     0.0000

#### Covertype Dataset
Composite score: 0.7603
- fidelity      0.9765
- utility       0.8061
- diversity     0.6716
- privacy       0.6434
- consistency   0.7042
- stability     0.0000

#### Bank Marketing Dataset
Composite score: 0.7757
- fidelity      0.9823
- utility       0.9683
- diversity     0.8583
- privacy       0.4291
- consistency   0.4098
- stability     0.0000

