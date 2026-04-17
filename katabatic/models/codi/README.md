# CoDi: Co-evolving Contrastive Diffusion Models

CoDi (Co-evolving Contrastive Diffusion) is a state-of-the-art generative model for mixed-type tabular data synthesis.

## Paper

**CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis**  
Lee et al., ICML 2023  
[arXiv:2304.12654](https://arxiv.org/abs/2304.12654)

## Installation

```bash
poetry install --extras codi
```

## Usage

Use the ready-made example script as a starting point:

```bash
python benchmarks/examples/run_codi_adult.py
```

Or integrate directly:

```python
from katabatic.models.codi.models import CODI

model = CODI(n_steps=50, epochs=100, batch_size=256)
model.train(
    data_dir="benchmarks/splits/my_dataset",
    synthetic_dir="benchmarks/synthetic/my_dataset/codi",
    categorical_cols=["workclass", "education"],
    continuous_cols=["age", "fnlwgt"],
)
synthetic_df = model.sample(n_samples=1000)
```

## Key Features

- 🎯 Handles mixed-type tabular data (continuous + categorical)
- 🚀 State-of-the-art synthetic data quality
- 🔄 Preserves complex feature dependencies
- 📊 Excellent utility for downstream ML tasks
