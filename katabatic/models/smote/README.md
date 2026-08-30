# SMOTE Model

## Model Overview

Synthetic Minority Oversampling Technique (SMOTE) model used for synthetic tabular data generation. 

The SMOTE model itself is a resampling method for imbalanced classification problems, example fraud detection logs. In cases like network logs, the data would be imbalanced towards the legitimate network traffic data (non-fraud); with fraud classified data being the minority class. Training on data of this nature will cause biases and result in poor fraud identification. SMOTE provides a method to level the two classes of data. This is done not by boot strapping or duplicating but instead creating new synthetic data examples for the minority class.
Alternatively the model creates synthetic minority data sets through interpolation between the existing minority class sample and one of the nearest minority-classes neighbouring it. This process is implemented prior to the downstream classifier training, hence SMOTE is used for rebalancing the training data and excels in problem spaces where the data is imabalanced. 
SMOTE is only applicable on feature generation for numeric data types only. Other varities are available for categorical datatypes and mix datatypes, but the SMOTE model itself can only be used on numerical data. 


---

### Key Idea
SMOTE is used as a tabular synthetic data generator with the following model interface:
1. `train(...)` loads and stores the real training data and prepares the nearest-neighbour structure required for generation.
2. `sample(n)` generates `n` new synthetic rows.
3. The generated features and labels are returned and may be saved as:
   - `x_synth.csv`
   - `y_synth.csv`
   - `metadata.json`
---

### Research Paper
SMOTE is based on:

> N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer,  
> **“SMOTE: Synthetic Minority Over-sampling Technique,”**  
> *Journal of Artificial Intelligence Research*, Volume 16, pp. 321–357, 2002.  
> DOI: 10.1613/jair.953


### Parameters Kept from the Original Method
The synthetic-generation process retains the main concepts from the paper:
- nearest-neighbour-based sample generation;
- interpolation between observations from the same class;
- configurable number of neighbours;
- stochastic generation of synthetic feature values.

### Project-Specific Interpretation

This tabular synthetic data generation of the SMOTE model is used within Katabatic, specifically for synethic data generation in situations where the sample data to replicated is numerical and imbalanced. Use cases such as financial fraud detection for example, situations where the overwhelming volume of data exceeds the minority class. SMOTE excels in the situations due to the prior imbalance process it applied prior to engaging in the synthetic data generation.

---

## Approach
The synthetic data generation pipeline is:

1. Load real training features `X`.
2. Load real target labels `y`.
3. Apply the required preprocessing.
4. Split the training observations by class.
5. Build a nearest-neighbour structure within each class.
7. Randomly select a source observation from that class.
8. Then randomly choose one of its nearest neighbours in the same class.
9. Interpolate between the two feature vectors.
10. Assign the source class label to the generated row.
11. Repeat process till the number of synthetic data records has been attained.

---

### Convergence Criteria

SMOTE has no convergence criterion.

---

## Hyperparameters
The exact hyperparameters depend on the repository implementation.

Typical SMOTE generation parameters include:

| Hyperparameter | Typical Value | Description |
|---|---:|---|
| `k_neighbors` | `5` | Number of same-class nearest neighbours available when choosing an interpolation partner. |
| `random_state` | `None` | Controls stochastic sample selection and interpolation for reproducibility. |
| `sampling_strategy` | Project-specific | Determines which target classes are selected during generation and in what proportions. |
| `n_samples` | Runtime argument | Number of synthetic observations requested from `sample(n)`. |

### Expected Files
Following the project template:

- `x_train.csv`
- `y_train.csv`

Or alternatively:

- `train_full.csv`

where the final column contains the target label.

---

### Numerical Features
SMOTE is applicable for use to create synthetic tabular data generation.

---

### Categorical Features
The SMOTE model can only be used against numerical data.
For datasets containing categorical columns, the following SMOTE variants can be used:

- `SMOTENC` - hybrid numerical and categorical data;
- `SMOTEN` - categorical only.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Strengths
- Simplicity in the implemention for synethic tabular data generation.
- Low compute requirements, example no GPU needed. 
- Able to generate synthetic data based on imbalanced datasets. 

---

## Limitations
- Limited to numerical data only.
- Lacks the capability of high computational deep learning models.
- Can be misused as the data created will have an over representation of the minority classes in datasets due to the imbalance process.

---

## Installation

Refer to the makefile. 

---

## Usage

### Project Model Interface

```python
from katabatic.models.smote.models import SMOTEModel

model = SMOTEModel(
    k_neighbors=5,
    sampling_strategy="auto",
    random_state=42,
)

model.train(
    data_dir=paths["split_dir"],
    synthetic_dir=paths.get("synthetic_dir"),
)

X_synth, y_synth = model.sample(1000)
```

---

### Evaluation Pipeline Benchmark Scripts

Evaluation pipeline Benchmark Scripts for each dataset:
- Magic [benchmarks/examples/MODEL/smote/magic](benchmarks/examples/smote/run_smote_magic.py)
- Shuttle [benchmarks/examples/MODEL/smote/shuttle](benchmarks/examples/smote/run_smote_shuttle.py)

---

## Model Evaluation Benchmarks Results

### Magic Dataset

Composite score: `TBD`

Dimension scores:

- fidelity: `TBD`
- utility: `TBD`
- diversity: `TBD`
- privacy: `TBD`
- consistency: `TBD`
- stability: `TBD`

### Shuttle Dataset

Composite score: `TBD`

Dimension scores:

- fidelity: `TBD`
- utility: `TBD`
- diversity: `TBD`
- privacy: `TBD`
- consistency: `TBD`
- stability: `TBD`

---

## Model Performance Benchmarks Results

TBC

---

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). **SMOTE: Synthetic Minority Over-sampling Technique.** *Journal of Artificial Intelligence Research, 16*, 321–357. DOI: 10.1613/jair.953.
2. `imbalanced-learn` documentation for `SMOTE`, `SMOTENC`, and `SMOTEN`.
