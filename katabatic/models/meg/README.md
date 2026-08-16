# MEG: Masked Ensemble Generator

Production-level implementation of MEG for the Katabatic framework.

## Paper

**MEG: Masked Ensemble Tabular Data Generator**  
Zhang et al.  
[Pre-publication Draft](https://www.nayyarzaidi.com/papers/MEG_Pre-publication_Draft.pdf)


## Overview

MEG is a generative model for tabular data that learns feature dependencies using a masked reconstruction strategy combined with an ensemble of neural networks.

Instead of relying on explicit probabilistic models, MEG captures relationships by iteratively masking and reconstructing feature subsets, enabling robust synthetic tabular data generation.

This implementation is adapted for Katabatic and supports standard synthetic data benchmarking workflows, including Train Synthetic Test Real (TSTR).

## Model Type
Ensemble-Based Generative Model


## Methodology
MEG generates synthetic data using a masked learning strategy:
- Random subsets of features are masked during training
- Multiple neural networks are trained to reconstruct missing values
- An ensemble of models improves robustness and diversity
- Synthetic samples are generated through iterative imputation

This approach enables the model to capture complex relationships in tabular data without relying on explicit probabilistic assumptions.


## Installation

```bash
pip install numpy pandas torch
```

---

## Usage

```python
from katabatic.models.meg import MEGModel
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

pipeline = TrainTestSplitPipeline(model=MEGModel)
pipeline.run(
    input_csv='data/adult.csv',
    output_dir='sample_data/adult',
    synthetic_dir='synthetic/adult/meg',
    real_test_dir='sample_data/adult'
)
```

## Key Features

- Masked reconstruction-based learning
- Ensemble of neural generators
- Supports mixed tabular data with categorical and numerical features
- Schema-based encoding and decoding
- Iterative imputation for realistic synthetic samples
- Compatible with Katabatic TSTR benchmarking workflows

---

## Architecture

```text
Input Data
    ↓
Schema-Based Encoding
    ↓
Masked Feature Selection
    ↓
Ensemble of Reconstruction Networks
    ↓
Iterative Imputation
    ↓
Decoded Synthetic Data
```


## How It Works

### Training Phase

1. Load `x_train.csv` and `y_train.csv`
2. Infer the schema of the training data
3. Encode categorical features using one-hot encoding
4. Convert numerical features into float representation
5. Randomly mask feature spans during training
6. Train an ensemble of masked reconstruction networks
7. Learn feature dependencies using reconstruction loss

### Generation Phase

1. Sample seed rows from the training distribution
2. Apply random masks to selected feature spans
3. Reconstruct masked values using the trained ensemble
4. Repeat the process through iterative imputation
5. Apply categorical hardening to keep categorical outputs valid
6. Decode the generated data back to tabular format


## Output

The model generates the following files:

- `x_synth.csv` - Synthetic feature data
- `y_synth.csv` - Synthetic labels
- `synthetic.csv` - Combined synthetic dataset


## Project Structure

```text
meg/
├── __init__.py
├── models.py
└── utils.py
```


## Main Components

### `models.py`

Contains the core MEG implementation, including:

- `MaskedNet`
- `MEGModel`
- Training logic
- Sampling logic
- Iterative masked imputation

### `utils.py`

Contains preprocessing utilities, including:

- Schema inference
- Categorical encoding
- Numerical conversion
- Decoding generated data
- Feature span creation for masking

### `__init__.py`

Exposes the MEG model for framework-level imports.


## Notes

- Categorical features are handled using one-hot encoding.
- Numerical features are converted into float values.
- Hard one-hot constraints are applied to categorical blocks during generation.
- Ensemble diversity improves stability and synthetic data quality.
- Evaluation is handled through the Katabatic benchmarking pipeline.


## Citation

```bibtex
@article{zhang2024meg,
  title={MEG: Masked Ensemble Tabular Data Generator},
  author={Zhang, Yishuo and Zaidi, Nayyar A. and Li, Gang and Buntine, Wray},
  year={2024}
}
```
