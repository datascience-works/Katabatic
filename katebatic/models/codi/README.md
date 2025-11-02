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

```python
from katebatic.models.codi import CODI
from katebatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

pipeline = TrainTestSplitPipeline(model=CODI)
pipeline.run(
    input_csv='data/my_dataset.csv',
    output_dir='sample_data/my_dataset',
    synthetic_dir='synthetic/my_dataset/codi',
    real_test_dir='sample_data/my_dataset'
)
```

See `examples/codi.ipynb` for more examples.

## Key Features

- 🎯 Handles mixed-type tabular data (continuous + categorical)
- 🚀 State-of-the-art synthetic data quality
- 🔄 Preserves complex feature dependencies
- 📊 Excellent utility for downstream ML tasks
