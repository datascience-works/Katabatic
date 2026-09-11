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
from katabatic.models.medgan import MEDGAN
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

pipeline = TrainTestSplitPipeline(model=MEDGAN)
pipeline.run(
    input_csv='data/adult.csv',
    output_dir='sample_data/adult',
    synthetic_dir='synthetic/adult/medgan',
    real_test_dir='sample_data/adult'
)
```

See `examples/medgan.ipynb` for more examples.

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

## Updated - 2026.09.07

The MedGAN implementation has been validated with the main datasets currently available in the Katabatic project, including Car, Nursery, Adult, MAGIC, and Shuttle.

A comparison was also completed between the Katabatic datasets and the datasets used in the original MedGAN research paper.

The original MedGAN paper focuses on high-dimensional discrete binary and count-based medical records. Among the datasets currently available in Katabatic, Nursery was identified as the closest structural match because it is fully categorical/discrete and contains no continuous numerical features.

Car was identified as the second closest match because it is also fully categorical.

Current structural matching order:

1. Nursery – closest available match
2. Car – strong discrete-data match
3. Adult – mixed categorical and numerical features
4. MAGIC – mainly numerical features
5. Shuttle – numerical features

The MedGAN implementation is also integrated with the Katabatic evaluation pipeline and supports evaluation across Fidelity, Utility, Diversity, Privacy, Consistency, and Stability.

The Katabatic datasets are not direct replacements for the medical datasets used in the original MedGAN research paper; the comparison is based on dataset structure and feature type.
