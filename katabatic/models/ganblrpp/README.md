# GANBLR++ Model

## Model Overview

GANBLR++ extends GANBLR by adding support for numerical attributes in tabular datasets. The original GANBLR model is mainly designed for categorical data, while GANBLR++ enables the model to work with datasets containing both numerical and categorical features.

---

### Key Idea

GANBLR++ extends the GANBLR framework by incorporating numerical attribute handling while retaining the Bayesian Network based generative approach.

For numerical columns, the implementation uses `DMMDiscretizer` before model training. After synthetic data is generated, the numerical features are converted back to their original numerical form.

The GANBLR++ research paper also proposes the use of a Dirichlet Mixture Model and unrestricted Bayesian Networks to improve numerical data generation and model flexibility. 

---

### Research Paper

GANBLR++ is based on:

Zhang, Y., Zaidi, N. A., Zhou, J., & Li, G. (2022).  
**GANBLR++: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks.**  
Proceedings of the 2022 SIAM International Conference on Data Mining (SDM), pp. 298–306.

DOI: `10.1137/1.9781611977172.34`

The paper introduces GANBLR++ as an extension of GANBLR with the ability to generate numerical attributes. 
---

## Approach

GANBLR++ follows the existing GANBLR workflow while adding preprocessing and postprocessing support for numerical features.

The main steps are:

1. Receive the input tabular data and labels.
2. Identify the numerical columns provided to the model.
3. Discretize numerical features using `DMMDiscretizer`.
4. Train the underlying GANBLR model on the processed data.
5. Generate synthetic samples.
6. Convert the generated numerical columns back to their original numerical representation.

### Training Details

The model inherits the existing GANBLR training process.

GANBLR uses a GAN-based framework together with Bayesian Networks for tabular data generation. GANBLR++ adds numerical-data handling around this existing training process.

### Convergence Criteria

GANBLR++ follows the convergence and stopping behaviour of the underlying GANBLR implementation. Training is controlled by the GANBLR training parameters such as the number of epochs.

---

## Hyperparameters

GANBLR++ uses the existing GANBLR hyperparameters together with the numerical-column configuration.

Important parameters include:

- `numerical_columns`: List of numerical columns that require GANBLR++ numerical processing.
- `k`: Controls the Bayesian Network dependency structure used by GANBLR.
- `epochs`: Number of training epochs.
- `batch_size`: Number of samples processed in each training batch.

---

## Input

The model accepts:

- `X`: Tabular feature matrix.
- `y`: Target labels.

### Expected Files

The Katabatic pipeline commonly uses:

- `x_train.csv`
- `y_train.csv`

Alternatively, a complete training dataset can be provided with the target label stored as the final column.

---

## Preprocessing

GANBLR++ introduces additional preprocessing for numerical features while using the existing GANBLR processing pipeline for the remaining data.

### Numerical Features

Numerical features are processed using `DMMDiscretizer` before training.

The discretized representation allows the GANBLR model to work with numerical columns during its categorical-style training process.

After synthetic samples are generated, the numerical columns are converted back into their original numerical representation.

### Categorical Features

Categorical features continue to use the existing GANBLR processing behaviour.

The GANBLR model works naturally with categorical/discrete tabular features, so these columns do not require the additional numerical conversion used by GANBLR++.

---

## Label Handling

Target labels are supplied separately through `y` during training.

The GANBLR++ implementation follows the existing GANBLR label-handling process.

---

## Output

GANBLR++ generates synthetic tabular data containing both categorical and supported numerical features.

Typical Katabatic pipeline outputs include:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Evaluation

GANBLR++ synthetic data can be evaluated using the Katabatic evaluation pipeline.

The Katabatic framework supports Train on Synthetic, Test on Real (TSTR) evaluation and other synthetic-data quality metrics. 

---

## Strengths

- Extends GANBLR to support numerical attributes.
- Supports mixed tabular datasets containing categorical and numerical features.
- Reuses the existing GANBLR implementation.
- Numerical features are converted back to numerical form after synthetic-data generation.
- Retains the Bayesian Network based GANBLR approach.

---

## Limitations

- Numerical attributes require additional preprocessing and postprocessing.
- Model performance can depend on the selected GANBLR training parameters.
- Training GAN-based models can require more computational time than simpler statistical methods.
- Generated-data quality can vary depending on the characteristics of the dataset.

---

## Installation

Install the GANBLR dependencies using Poetry:

```bash
poetry install -E ganblr
```

---

## Usage

```python
from katabatic.models.ganblrpp import GANBLRPP

model = GANBLRPP(
    numerical_columns=["column_name"]
)

model.fit(X, y)

synthetic_data = model.sample(size=100)
```

---

## Model Evaluation Benchmarks Results

GANBLR++ can be evaluated using the Katabatic benchmark pipeline.

Benchmark results should be added here after running the model on the required datasets.

---

## Model Performance Benchmarks Results

Runtime and performance depend on the dataset size, hardware configuration, and training parameters.

Measured benchmark runtimes should be added here when available.
