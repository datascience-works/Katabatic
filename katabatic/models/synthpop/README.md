# SynthPop for Katabatic Repository

SynthPop implementation for synthetic tabular data generation using CART-based sequential conditional synthesis via the R synthpop package.

## Overview

This SynthPop implementation is designed specifically for:
- Synthetic tabular data generation for benchmarking (TSTR evaluation)
- Classification and regression datasets
- Privacy-preserving synthesis using CART models
- Integration into Katabatic's standard pipeline outputs (x_train.csv, y_train.csv)

SynthPop is a statistical disclosure control method. It generates synthetic data by sequentially modelling each column conditioned on all previously synthesized columns using Classification and Regression Trees (CART). The synthesis is performed entirely via the official R synthpop package, called from Python using subprocess.

## Paper and Implementation Reference

### Research Paper

**synthpop: Bespoke Creation of Synthetic Data in R**
Beata Nowok, Gillian M. Raab, Chris Dibben
University of Edinburgh
Journal of Statistical Software, Volume 74, Issue 11, 2016

Paper link: https://doi.org/10.18637/jss.v074.i11

### Implementation Reference

The implementation uses the official R synthpop package available on CRAN:
https://CRAN.R-project.org/package=synthpop

The Python wrapper has been adapted into Katabatic's required 3-file model structure.

## Model Type

- Category: Synthetic Tabular Data Generator
- Approach: Sequential conditional synthesis using CART
- Synthesis Method: Classification and Regression Trees (CART) via R synthpop
- Language: Python wrapper calling R via subprocess
- Supports:
  - Classification
  - Regression

## Features

- CART-based sequential synthesis preserving variable relationships
- Handles mixed data types (numerical, categorical, ordinal)
- No neural training, no epochs
- Privacy-preserving by design (no direct record copying)
- Works with Katabatic train/test split pipeline outputs
- Saves synthetic outputs in standard CSV format

## Repository Files

- models.py
  Katabatic wrapper class that:
  - accepts input CSV path (full training data, X + y)
  - writes and executes an R script via subprocess
  - saves synthetic output CSV to the specified path

- utils.py
  Helper functions for:
  - writing the R script that calls the synthpop package (write_r_script)

- __init__.py
  Exposes SynthPop for clean imports in Katabatic.

## Installation

### Requirements

Install Python dependencies:

```
pip install pandas
```

Install R:
- Download from https://www.r-project.org/
- Ensure Rscript is available on your system PATH

Install the R synthpop package (run once in R console):

```r
install.packages("synthpop")
```

### Verifying R Installation

To verify Rscript is accessible from Python:

```
Rscript --version
```

### Colab Installation

If running in Google Colab:

```python
import subprocess
subprocess.run(["apt-get", "install", "-y", "r-base"], check=True)
subprocess.run(["Rscript", "-e", "install.packages('synthpop', repos='https://cloud.r-project.org/')"], check=True)
```

## Quick Start

### Basic Usage (Standalone Concept)

1) Prepare training split data:
- x_train.csv
- y_train.csv

2) Run generator (conceptual example):

```python
from katabatic.models.synthpop import SynthPop
import pandas as pd

model = SynthPop(seed=42)

model.train(
    dataset_path="path/to/train_full.csv",
    synthetic_path="path/to/synthetic_full.csv"
)
```

Output:
- synthetic_full.csv: full synthetic table in original column order

## Integration With Katabatic Pipeline

SynthPop is designed to plug into Katabatic's pipeline that outputs train split CSVs.

Expected input structure:
- output_dir/
  - x_train.csv
  - y_train.csv

Expected synthetic outputs:
- synthetic_dir/
  - synthetic.csv
  - x_synth.csv
  - y_synth.csv

The adapter merges x_train and y_train before passing to SynthPop, then splits the synthetic output back into x_synth and y_synth.

Example pipeline usage (typical Katabatic flow):

```python
from katabatic.models.synthpop.adapter import SynthPopAdapter

adapter = SynthPopAdapter(seed=42)

adapter.train(
    dataset_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/synthpop",
    label_col="income"
)
```

## Parameter Guide

### Core SynthPop Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| seed | 42 | Random seed passed to R set.seed() and syn() for reproducibility |
| method | "cart" | Synthesis method used by synthpop (CART by default, hardcoded in R script) |

### Practical Tuning Suggestions

- If synthetic data quality is weak:
  - SynthPop CART is generally robust; quality issues usually relate to small datasets
  - For small datasets, consider increasing training data size before synthesis

- If generation is slow:
  - SynthPop is column-sequential; runtime scales with number of columns
  - Large datasets with many columns will take longer

## Training vs Generation Notes

SynthPop differs from GAN and diffusion-based models:
- There is no adversarial training loop
- There are no epochs or gradient updates
- Each column is synthesized sequentially, conditioned on all previously synthesized columns
- The first column is synthesized by random sampling with replacement from the original
- All subsequent columns use CART models fitted on the original data

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
3) Generate synthetic training data using SynthPop
4) Run TSTR evaluation:
   - Train classifier on synthetic train
   - Test on real test
5) Log metrics:
   - Accuracy
   - F1-score
   - AUC (if binary / supported)

## Results Reporting

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

If you run SynthPop in Google Colab:
- Install R and the synthpop package first (see Installation section above)
- First run may take longer while R packages are installed
- CPU runtime is sufficient; GPU is not required

If Colab disconnects:
- Save outputs to Google Drive or download synthetic CSV outputs after run

## Troubleshooting

### Issue: Rscript not found
Solution:
- Ensure R is installed and Rscript is on your system PATH
- On Colab: run apt-get install r-base first

### Issue: synthpop package not found in R
Solution:
- Open R console and run: install.packages("synthpop")
- On Colab: run the Rscript installation command shown in the Installation section

### Issue: Synthesis fails on a specific column
Solution:
- Check for constant columns (zero variance) in the training data
- Check for columns with only NA values
- Ensure the label column name matches what is passed to the adapter

### Issue: Output files not saved correctly
Solution:
- Ensure synthetic_dir exists or is created before running
- Check you have write permissions (Colab vs local paths)

## Citation

If using this implementation in reports or publications:

```
@article{nowok2016synthpop,
  title={synthpop: Bespoke Creation of Synthetic Data in R},
  author={Nowok, Beata and Raab, Gillian M. and Dibben, Chris},
  journal={Journal of Statistical Software},
  volume={74},
  number={11},
  year={2016},
  doi={10.18637/jss.v074.i11}
}
```
