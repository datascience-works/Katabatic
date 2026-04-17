# MedGAN: Medical Generative Adversarial Network

Production-level implementation of MedGAN for the Katabatic framework.

## Paper

**Generating Multi-label Discrete Patient Records using Generative Adversarial Networks**  
Edward Choi et al., MLHC 2017  
[arXiv:1703.06490](https://arxiv.org/abs/1703.06490)

## Overview

MedGAN uses a three-component architecture:

1. **Autoencoder** - Compresses high-dimensional binary/count data
2. **Generator** - Generates synthetic data in the latent space
3. **Discriminator** - Distinguishes real from synthetic representations

## Installation

```bash
poetry install --extras medgan
```

## Usage

```python
from katabatic.models.medgan.models import MEDGAN

model = MEDGAN()
model.train(
    data_dir="benchmarks/splits/my_dataset",
    categorical_cols=["workclass", "education"],
    continuous_cols=["age", "fnlwgt"],
)
synthetic_df = model.sample(n_samples=1000)
```

## Key Features

- 🏥 Designed for binary/count medical records
- 🎯 Works with any high-dimensional tabular data
- 🚀 Two-phase training (AE pretraining + GAN)
- 📊 Excellent for sparse, high-dimensional data

## Architecture

```
Input Data
    ↓
Autoencoder (Pretrained)
    ↓ (encode to latent space)
Latent Representation
    ↑
Generator ← Noise
    ↓
Discriminator (Real vs Fake)
```

## Citation

```bibtex
@inproceedings{choi2017generating,
  title={Generating multi-label discrete patient records using generative adversarial networks},
  author={Choi, Edward and Biswal, Siddharth and Malin, Bradley and Duke, Jon and Stewart, Walter F and Sun, Jimeng},
  booktitle={Machine Learning for Healthcare Conference},
  pages={286--305},
  year={2017}
}
```
