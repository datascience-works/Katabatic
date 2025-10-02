MedGAN

This folder contains the implementation of MedGAN (Choi et al., 2017) integrated into the Katabatic framework for benchmarking tabular generative models.

MedGAN is a Generative Adversarial Network (GAN) designed to generate multi-label discrete records, originally developed for healthcare data (e.g., diagnosis and procedure codes). Within Katabatic, it has been adapted for general tabular datasets such as the Adult dataset.

Key Features

Supports training on tabular datasets processed into patient-by-code–like matrices (binary/count representations).

Integrated with Katabatic’s fit(), sample(), and evaluate() workflow.

Produces synthetic data aligned with Katabatic’s benchmarking and evaluation pipeline.

Includes example notebooks for both Adult dataset and general usage.

File Structure

medgan.py – Core model implementation.

preprocess.py – Data preprocessing utilities.

driver.py – Driver script for standalone runs.

example.ipynb – General example notebook (training, sampling, evaluation).

Adult Example.ipynb – Demonstration notebook using the sample_data/adult dataset.

sample_data/ – Example dataset (Adult).

##Usage
###1. Import & Initialize
```python
from katebatic.models.medgan.medgan import Medgan

medgan = Medgan(
    repo_root=".",  # Path to the repository root
    epochs=100,
    batch_size=64,
    latent_dim=128
)
2. Fit the model
X_train, y_train = ... # Load processed training data
medgan.fit(X_train, y_train)

3. Generate synthetic samples
synthetic_data = medgan.sample(n=1000)

4. Evaluate via Katabatic

from katebatic.evaluate import evaluate_model

results = evaluate_model(real_data=X_train, synthetic_data=synthetic_data)
print(results)


Example Notebooks

Adult Example.ipynb → Run MedGAN on the Adult dataset provided in sample_data/adult.

example.ipynb → General template notebook for running MedGAN on any new dataset.

Integration Notes

MedGAN here is integrated within Katabatic’s model registry (registry.py).

To add a new dataset:

Place the dataset in sample_data/your_dataset.

Preprocess into matrix form (binary/count).

Update paths in your notebook or driver script.

Reference

Edward Choi, Siddharth Biswal, Bradley Malin, Jon Duke, Walter F. Stewart, Jimeng Sun.
Generating Multi-label Discrete Patient Records using Generative Adversarial Networks.
MLHC 2017.[link](https://arxiv.org/abs/1703.06490)