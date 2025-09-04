# TabDDPM: Modelling Tabular Data with Diffusion Models
This is the official code for our paper "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421))

<!-- ## Results
You can view all the results and build your own tables with this [notebook](notebooks/Reports.ipynb). -->

## Setup the environment
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate tddpm
    ```

## Running the experiments
Follow the code from `tabddpm_example.ipynb`

```python
pipeline = TabDDPMPipeline(
    csv_path="/mnt/d/my_katabatic/raw_data/diabetes.csv",
    target_column="Outcome",
    normalization="quantile",
    train_overrides={"steps": 5000, "lr": 1e-3, "batch_size": 1024},
)
pipeline.fit()

synthetic_df = pipeline.sample(num_samples = 5000, batch_size = 1024)
```
