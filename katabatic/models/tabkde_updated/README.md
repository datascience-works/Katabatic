# TabKDE for Katabatic Repository

TabKDE implementation for synthetic tabular data generation using Kernel Density Estimation with copula transformations.

## Overview

This TabKDE implementation is designed specifically for:
- Synthetic tabular data generation for benchmarking (TSTR evaluation)
- Classification and regression datasets
- Fast, scalable generation with no neural training required
- Integration into Katabatic's standard pipeline outputs (x_train.csv, y_train.csv)

TabKDE is a statistical density-based generation approach. It maps tabular data into a copula-transformed latent space, learns the Distance to Closest Record (DCR) distribution via a Gaussian Mixture Model, and generates synthetic samples by perturbing training points with GMM-sampled radii and random directions. There is no adversarial training, no epochs, and no GPU required.

## Paper and Implementation Reference

### Research Paper

**TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates**

Paper link: https://arxiv.org/abs/2605.17642

### Implementation Reference

The implementation follows the methodology described in the paper and has been adapted into Katabatic's required 3-file model structure.

Official GitHub repository: https://github.com/tabkde/tabkde-main

## Model Type

- Category: Synthetic Tabular Data Generator
- Approach: Statistical density-based generation (KDE with copula transformations)
- Sampling: GMM-calibrated KDE with empirical DCR distribution
- Supports:
  - Classification
  - Regression

## Features

- No neural training, no epochs, no GPU required
- Works with Katabatic train/test split pipeline outputs
- Handles mixed data types (numerical, categorical)
- Encodes categoricals via category codes before copula transform
- Learns DCR distribution via Gaussian Mixture Model (BIC-selected components)
- Generates synthetic samples by perturbing latent points with GMM-sampled radii
- Clips all samples to [0,1] copula space to respect marginal boundaries
- Saves synthetic outputs in standard CSV format

## Repository Files

- models.py
  Core TabKDE model class that:
  - preprocesses the full training table (X + y)
  - fits the empirical copula transformer
  - learns the DCR GMM distribution
  - generates synthetic samples via KDE sampling
  - inverse transforms back to original feature space

- utils.py
  Helper functions for:
  - array shape handling (ensure_2d)
  - DCR distance computation (compute_min_distances_cpu)
  - data preprocessing and categorical encoding (preprocess_data)
  - empirical copula transformation (EmpiricalTransformer)
  - KDE-style sampling via GMM distances (sample_points_via_dcp_distribution)

- __init__.py
  Exposes TabKDEModel for clean imports in Katabatic.

## Installation

### Requirements

Install core dependencies:

```
pip install numpy pandas scikit-learn scipy
```

No additional model-specific packages required. TabKDE runs entirely on CPU using standard scientific Python libraries.

## Quick Start

### Basic Usage (Standalone Concept)

1) Prepare training split data:
- x_train.csv
- y_train.csv

2) Run generator (conceptual example):

```python
from katabatic.models.tabkde import TabKDEModel
import pandas as pd

x_train = pd.read_csv("path/to/x_train.csv")
y_train = pd.read_csv("path/to/y_train.csv").iloc[:, 0]

model = TabKDEModel(
    n_dcr_splits=10,
    max_gmm_components=10,
    noise_std=0.01,
    random_state=42
)

model.fit(x_train, y_train)
synth_df, _ = model.sample(n_samples=len(x_train))
```

Output:
- synth_df: full synthetic table in original column order

## Integration With Katabatic Pipeline

TabKDE is designed to plug into Katabatic's pipeline that outputs train split CSVs.

Expected input structure:
- output_dir/
  - x_train.csv
  - y_train.csv

Expected synthetic outputs:
- synthetic_dir/
  - synthetic.csv
  - x_synth.csv
  - y_synth.csv

Example pipeline usage (typical Katabatic flow):

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tabkde import TabKDEModel

pipeline = TrainTestSplitPipeline(
    model=lambda: TabKDEModel(
        n_dcr_splits=10,
        max_gmm_components=10,
        noise_std=0.01,
        random_state=42
    ),
    input_csv="raw_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/tabkde"
)

pipeline.run()
```

## Parameter Guide

### Core TabKDE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_dcr_splits | 10 | Number of random splits to estimate the DCR distribution (higher = more stable GMM) |
| max_gmm_components | 10 | Maximum Gaussian components for GMM (BIC selects best k from 1 to this value) |
| noise_std | 0.01 | Scale of Gaussian noise added during sampling (increases diversity) |
| random_state | None | Random seed for reproducibility |

### Practical Tuning Suggestions

- If synthetic data quality is weak:
  - Increase n_dcr_splits (e.g., 10 → 20) for a more stable DCR distribution
  - Increase noise_std slightly (0.01 → 0.05) for more diversity
  - Increase max_gmm_components if the DCR distribution is multi-modal

- If generation is slow on large datasets:
  - Reduce n_dcr_splits (10 → 5)
  - TabKDE is CPU-only and already very fast; typical fit time is under 2 minutes

## Training vs Generation Notes

TabKDE differs from GAN and diffusion-based models:
- There is no adversarial training loop
- There are no epochs or gradient updates
- Fit time is dominated by the DCR GMM fitting (random splits + KDTree queries)
- Generation time is fast: sampling from GMM + inverse copula transform
- The full pipeline (fit + sample) typically completes in under 3 minutes on standard datasets

## Datasets Tested

This implementation has been tested on the following datasets within the Katabatic pipeline:
- Car
- Magic
- Shuttle
- Adult
- Nursery

## Recommended Workflow (Professional / Reproducible)

1) Preprocess dataset (if Katabatic requires discretization)
2) Run train/test split pipeline
3) Generate synthetic training data using TabKDE
4) Run TSTR evaluation:
   - Train classifier on synthetic train
   - Test on real test
5) Log metrics:
   - Accuracy
   - F1-score
   - AUC (if binary / supported)

## Results Reporting (Add This To Your Repo)

Create a RESULTS.md file and store:
- Dataset name
- Models tested (LR, MLP, RF, XGBoost)
- Metrics (Accuracy, F1, AUC)
- Notes (why performance improved or dropped)

Example format:

```
Dataset: Adult
- LR:  Acc=..., F1=..., AUC=...
- MLP: Acc=..., F1=..., AUC=...
- RF:  Acc=..., F1=..., AUC=...
- XGB: Acc=..., F1=..., AUC=...
```

## Colab Notes (Important)

If you run TabKDE in Google Colab:
- No special installations needed beyond the core dependencies
- No pretrained weights to download
- CPU runtime is sufficient; GPU is not required

If Colab disconnects:
- Save outputs to Google Drive or download synthetic CSV outputs after run

## Troubleshooting

### Issue: GMM fitting fails or returns None
Solution:
- Increase n_dcr_splits to get more DCR distance samples
- Check that training data has no constant columns (zero variance)
- Ensure x_train.csv and y_train.csv are not empty

### Issue: Synthetic data looks identical to training data
Solution:
- Increase noise_std (0.01 → 0.05 or higher)
- Increase max_gmm_components to capture multi-modal DCR distributions

### Issue: Categorical columns look wrong in synthetic output
Solution:
- Check that categorical columns are object or category dtype in input CSV
- preprocess_data encodes them as category codes; verify the original CSV is loaded correctly

### Issue: Output files not saved correctly
Solution:
- Ensure synthetic_dir exists or is created before running
- Check you have write permissions (Colab vs local paths)

## Citation

If using this implementation in reports or publications:

```
@software{tabkde_katabatic,
  title={TabKDE: Simple and Scalable Tabular Data Generation with Kernel Density Estimates (Katabatic Integration)},
  author={Rema Ramesh and Team},
  year={2026},
  url={GitHub repository URL}
}
```