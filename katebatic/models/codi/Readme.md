# CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis
This code is the official implementation of "CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis".
(https://arxiv.org/abs/2304.12654)

## Requirements
Run the following to install requirements:
```setup
conda env create --file environment.yaml
```

## Usage
`example.ipynb` shows how to generate synthetic data using CoDi architecture.
```python
synthetic_data = codi(
    csv_path='raw_data/iris.csv',
    test_split=0.2,
    total_epochs_both=20,
    training_batch_size=1024,
    num_samples=500,
)
```
Just need to pass in the dataset. Number of generated data is controlled by `num_sample`. 