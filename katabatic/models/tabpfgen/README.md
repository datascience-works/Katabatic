# TabPFGen for Katabatic Repository

TabPFGen implementation for synthetic tabular data generation using TabPFN-guided sampling.

## Overview

This TabPFGen implementation is designed specifically for:
- Synthetic tabular data generation for benchmarking (TSTR evaluation)
- Classification and regression datasets
- Fast, strong conditional guidance using TabPFN
- Integration into Katabatic’s standard pipeline outputs (x_train.csv, y_train.csv)

TabPFGen is an energy-based generation approach that uses TabPFN as a guiding model. Instead of training a GAN, it generates synthetic samples by iteratively updating candidate samples through SGLD-style sampling while being guided by TabPFN predictions.

## Paper and Implementation Reference

### Research Paper

**TabPFGen: Tabular Data Generation with TabPFN**  
Junwei Ma, Apoorv Dankar, George Stein, Guangwei Yu, Anthony Caterini  
Layer 6 AI, Toronto, Canada  

Paper link: https://arxiv.org/abs/2406.05216

### Implementation Reference

GitHub reference used during implementation:

https://github.com/sebhaan/TabPFGen

The implementation has been adapted into Katabatic’s required 3-file model structure.

## Model Type

- Category: Synthetic Tabular Data Generator
- Approach: Energy-based / guidance-based generation (TabPFN-guided)
- Guidance Model: TabPFN (pretrained)
- Sampling: Stochastic Gradient Langevin Dynamics (SGLD)
- Supports:
  - Classification (optionally balanced class generation)
  - Regression

## Features

- TabPFN-guided generation for tabular data
- Works with Katabatic train/test split pipeline outputs
- Supports CPU and GPU execution
- Saves synthetic outputs in standard CSV format
- Optionally generates balanced samples for imbalanced classification
- Clear separation between:
  - Core generator logic (core.py)
  - Katabatic wrapper interface (model.py)

## Repository Files


- model.py
  Katabatic wrapper class that:
  - reads x_train.csv / y_train.csv from pipeline output
  - calls TabPFGen core generator
  - writes synthetic.csv / x_synth.csv / y_synth.csv

- utils.py
  Helper functions for:
  - detecting target column naming
  - splitting feature and label arrays cleanly

- __init__.py
  Exposes TabPFGenModel / TabPFGen wrapper for clean imports in Katabatic.

## Installation

### Requirements

Install core dependencies:

pip install numpy pandas scikit-learn torch

Install TabPFN:

pip install tabpfn

Important note:
- On the first run, TabPFN may automatically download pretrained weights/checkpoints.
- This is expected behaviour (especially on Google Colab).
- If you are running in Colab, it may prompt or take time during the first download.

## Quick Start

### Basic Usage (Standalone Concept)

1) Prepare training split data:
- x_train.csv
- y_train.csv

2) Run generator (conceptual example):

from tabpfgen import TabPFGenModel

model = TabPFGenModel(
    task="classification",
    n_sgld_steps=300,
    sgld_step_size=0.01,
    sgld_noise_scale=0.01,
    balance_classes=True,
    device="auto"
)

model.fit(train_dir="path/to/train_split_folder")
model.generate(save_dir="path/to/output_folder")

Output files:
- synthetic.csv (full combined synthetic data)
- x_synth.csv (features only)
- y_synth.csv (labels only)

## Integration With Katabatic Pipeline

TabPFGen is designed to plug into Katabatic’s pipeline that outputs train split CSVs.

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

from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.tabpfgen import TabPFGenModel

pipeline = TrainTestSplitPipeline(
    model=lambda: TabPFGenModel(
        task="classification",
        n_sgld_steps=300,
        sgld_step_size=0.01,
        sgld_noise_scale=0.01,
        balance_classes=True,
        device="auto"
    ),
    input_csv="raw_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/tabpfgen"
)

pipeline.run()

## Parameter Guide

### Core TabPFGen Parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| task | "classification" | "classification" or "regression" |
| n_sgld_steps | 300 | Number of SGLD sampling steps (higher = more refinement, slower) |
| sgld_step_size | 0.01 | Step size for sample updates |
| sgld_noise_scale | 0.01 | Noise added during sampling (prevents collapse, improves diversity) |
| balance_classes | True | If classification: generate class-balanced samples |
| device | "auto" | "auto", "cpu", or "cuda" |
| y_col | auto | Output label column name (commonly "class" or "target") |

### Practical Tuning Suggestions

- If synthetic data quality is weak:
  - Increase n_sgld_steps (e.g., 300 → 500)
  - Increase sgld_noise_scale slightly (0.01 → 0.02)
  - Ensure TabPFN is running on GPU in Colab if possible

- If generation is too slow:
  - Reduce n_sgld_steps (300 → 150)
  - Run on GPU ("cuda")

## Training vs Generation Notes

TabPFGen differs from GAN-based models:
- There is no adversarial training loop
- The main cost is iterative sample refinement during generation
- TabPFN acts like a strong guiding model for shaping synthetic samples

## Recommended Workflow (Professional / Reproducible)

1) Preprocess dataset (if Katabatic requires discretization)
2) Run train/test split pipeline
3) Generate synthetic training data using TabPFGen
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

Dataset: Adult
- LR: Acc=..., F1=..., AUC=...
- MLP: Acc=..., F1=..., AUC=...
- RF:  Acc=..., F1=..., AUC=...
- XGB: Acc=..., F1=..., AUC=...

## Colab Notes (Important)

If you run TabPFGen in Google Colab:
- You must install tabpfn (pip install tabpfn)
- First run may download pretrained TabPFN weights
- Use GPU runtime for faster sampling:
  Runtime → Change runtime type → GPU

If Colab disconnects:
- Save outputs to Google Drive or download synthetic CSV outputs after run

## Troubleshooting

### Issue: “TabPFN download” appears or takes long
Solution:
- This is expected the first time.
- Let it finish once; future runs usually reuse cached weights.

### Issue: Slow generation
Solution:
- Use GPU runtime (Colab GPU)
- Reduce n_sgld_steps temporarily
- Use smaller batch sizes if implemented

### Issue: Synthetic labels look wrong (classification)
Solution:
- Ensure balance_classes is correctly configured
- Confirm y_train.csv labels are correct and clean
- Check target column inference in utils.py (or pass y_col explicitly)

### Issue: Output files not saved correctly
Solution:
- Ensure synthetic_dir exists or is created
- Check you have write permissions (Colab vs local paths)

## Suggested Extra Files To Add (Professional Repo Look)

- requirements.txt
- notebooks/tabpfgen_colab_demo.ipynb
- scripts/run_tabpfgen_pipeline.py
- RESULTS.md
- CONTRIBUTING.md (optional)
- LICENSE

## Citation

If using this implementation in reports or publications:

@software{tabpfgen_katabatic,
  title={TabPFGen: TabPFN-Guided Synthetic Tabular Data Generation (Katabatic Integration)},
  author={Rema Ramesh and Team},
  year={2026},
  url={GitHub repository URL}
}

## Support

For issues/questions:
- Use GitHub Issues in the repository
- Include:
  - dataset name
  - environment (local / colab)
  - error trace
  - parameter configuration

##  TabPFN License & Authentication (Important)

TabPFGen relies on the TabPFN library, which requires a **one-time license acceptance** to download pretrained weights.

### Steps to Set Up:

1. Visit: https://ux.priorlabs.ai  
2. Create or log in to your account  
3. Accept the license (Licenses tab)  
4. Copy your API key  
5. Set environment variable:

```python
import os
os.environ["TABPFN_TOKEN"] = "your_api_key_here"