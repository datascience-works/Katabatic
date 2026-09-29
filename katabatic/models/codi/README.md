# CoDi: Co-evolving Contrastive Diffusion Models

CoDi (Co-evolving Contrastive Diffusion) is a state-of-the-art generative model for mixed-type tabular data synthesis.

## Overview

CoDi, a synthetic tabular data generator that allows for rows that mix continuous and categorical columns, motivated by the observation that prior methods which one-hot all categoricals into a single continuous model tend to break the correlations between numeric and discrete fields. Its core idea is to run two diffusion models in parallel a Gaussian DDPM for the continuous columns and a multinomial diffusion for the categorical columns. The two models are co-evolved: at every noising and denoising step, each model reads the other's current state as a condition, so the continuous denoiser sees the noisy categoricals and vice versa, which keeps cross-type correlations intact. CoDi adds a contrastive loss that pairs each real row with a positive prediction (conditioned on its true counterpart) and a negative one (conditioned on a randomly shuffled counterpart from another row), training each model to stay close to the matching condition and far from a mismatched one. At sampling time, both models run their reverse processes simultaneously, exchanging conditions step by step, and produce a synthetic row whose continuous and categorical halves were generated together rather than stitched.


<img src="image.png" alt="description" width="500" height="200" />


## Paper

**CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis**
Lee et al., ICML 2023
[arXiv:2304.12654](https://arxiv.org/abs/2304.12654)

## Implementation Details

**Repository**: Adapted from https://github.com/ChaejeongLee/CoDi

## References

**CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis**
Chaejeong Lee, Jayoung Kim, Noseong Park
*Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:18940–18956*
Paper: <https://proceedings.mlr.press/v202/lee23i.html>
Preprint: <https://arxiv.org/abs/2304.12654>
Code: <https://github.com/ChaejeongLee/CoDi>


## Installation

```bash
poetry install --extras codi
```

## Usage

```python
from katabatic.models.codi import CODI
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

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

***

---

## Usage
Evaluation pipeline Benchmark Scripts for each dataset:
- Adult [benchmarks/examples/codi/run_codi_adult](benchmarks/examples/codi)
- Shuttle [benchmarks/examples/codi/run_codi_shuttle](benchmarks/examples/codi)
- Car [benchmarks/examples/codi/run_codi_car](benchmarks/examples/codi)
- Magic [benchmarks/examples/codi/run_codi_magic](benchmarks/examples/codi)
- Nursery [benchmarks/examples/codi/run_codi_nursery](benchmarks/examples/codi)

```python

config = RunConfig(
    dataset_name="car",
    model_name="ganblr",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

model = CODI(n_steps=50, epochs=100, batch_size=256)
model.train(
    paths["split_dir"],
    paths["synthetic_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

```

---

## Model Evaluation Benchmarks Results

#### FINAL RESULTS - Cars Dataset
Composite score : 0.7250
Dimension scores:
  fidelity       0.8638
  utility        0.8131
  diversity      0.9721
  privacy        0.4057
  consistency    0.1785
  stability      0.9720

---

## Model Performance Benchmarks Results

#### ⏰ Evaluation Runtime Report 🧾
Start time: 114932.215904212
End time: 114948.459338917
CODI has taken 16.243434705000254 seconds to run the car dataset.

#### 💻 Computation Hardware Summary 🧾
  🖥️  System:     Linux
  🏠  Node:       GPU
  📦  Release:    6.6.87.2-microsoft-standard-WSL2
  🔢  Version:    #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025
  🔧  Processor:  x86_64
  🎮  GPU:        NVIDIA GeForce RTX 3060
  📟  Total RAM:  50.5164 GB
  💾  Free RAM:   48.2754 GB
  ⚡  Used RAM:   1.6924 GB

---
