# CTAB-GAN+ for Katabatic

## Model Type
Conditional Generative Adversarial Network (GAN) for synthetic tabular data generation.

## Model Overview
CTAB-GAN+ is an advanced conditional GAN framework designed to generate high-quality synthetic tabular data. It extends the original CTAB-GAN by improving the handling of mixed data types, skewed numerical distributions, rare categorical values, and imbalanced classes.

The model uses conditional vectors to guide the generation process and specialised preprocessing methods to represent categorical, continuous, mixed, and long-tail variables. CTAB-GAN+ also applies Wasserstein GAN training with gradient penalty (WGAN-GP) to improve stability and convergence during adversarial training.

Within Katabatic, CTAB-GAN+ is used to generate synthetic datasets in the standard format required for benchmarking and evaluation using the Train Synthetic Test Real (TSTR) protocol.

---

## Research Paper
Zhao, Z., Kunar, A., Birke, R. and Chen, L. Y. (2022)  
CTAB-GAN+: Enhancing Tabular Data Synthesis  
https://arxiv.org/abs/2204.00401

---

## Official GitHub Repository
https://github.com/Team-TUD/CTAB-GAN-Plus

---

## Implementation Details
This implementation is adapted from the official CTAB-GAN+ repository and has been refactored to align with the Katabatic framework.

Key adaptations:
- Converted into Katabatic 3-file structure
- Removed adapter-based design
- Integrated direct training interface
- Ensured compatibility with Katabatic pipelines
- Standardised input/output formats

No major architectural changes were made. The model follows the original training logic and design principles.

---

## Katabatic Model Structure

katabatic/models/ctabgan_plus/

- __init__.py → exposes model
- models.py → main model and training logic
- utils.py → preprocessing and transformations

---

## Dependencies

Install required packages before running:

pip install pandas numpy scikit-learn torch torchvision

---

## Dataset Format (Input)

sample_data/<dataset_name>/

- x_train.csv
- y_train.csv
- x_test.csv
- y_test.csv

---

## Output Format (Generated)

synthetic/<dataset_name>/ctabgan_plus/

- x_synth.csv
- y_synth.csv

---

## Datasets Used

- CAR
- MAGIC
- NURSERY
- ADULT
- SHUTTLE

## Epoch Configuration

CAR → 300 epochs  
MAGIC → 200 epochs  
NURSERY → 200 epochs  
ADULT → 150 epochs  
SHUTTLE → 150 epochs  

These values are selected based on dataset size and computational cost while maintaining convergence stability.


## Running the Model (Example)

from katabatic.models.ctabgan_plus.models import CTABGANPlus

model = CTABGANPlus(
    config={"epochs": 200}
)

model.train(
    dataset_dir="sample_data/magic",
    synthetic_dir="synthetic/magic/ctabgan_plus",
    categorical=[10]
)

Note:
- categorical must be passed as column indices
- label column should be included in categorical indices

---

## Pipeline Steps

1. Split dataset into train/test
2. Train CTAB-GAN+ on training data
3. Generate synthetic data
4. Encode categorical features for evaluation
5. Reindex labels for compatibility
6. Run TSTR evaluation


## Important Notes

- Works best on mixed-type tabular datasets
- Rare classes may not always be generated
- Label reindexing is required for XGBoost
- Requires correct categorical column indices
- Performance depends on dataset balance and preprocessing
