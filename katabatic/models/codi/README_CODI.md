# ARF Model

## Model Overview
CoDi (Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis) is a synthetic tabular data generation model designed for datasets containing both continuous and categorical variables.
CoDi uses two co-evolving diffusion models: one for continuous variables and one for discrete variables. The two models condition on each other during training and sampling to preserve the relationships between continuous and categorical features.

---

### Key Idea
CoDi models continuous and discrete variables in their appropriate spaces instead of converting all variables into the same continuous representation.

The continuous model uses Gaussian diffusion, while the discrete model uses categorical diffusion. The two diffusion models co-evolve by using the noisy representation of the other variable type as a condition.

The paper also introduces contrastive learning to strengthen the relationship between continuous and discrete variables by comparing positive and negative conditions during training.

---

### Research Paper
Lee, C., Kim, J., & Park, N..  
**CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis.**  
Paper: https://arxiv.org/abs/2304.12654 

---

## Approach
CoDi separates mixed-type tabular data into continuous and categorical variables and models them using two diffusion models. The main pipeline includes preprocessing the two data types, applying Gaussian diffusion to continuous variables and categorical diffusion to discrete variables, jointly training the two diffusion models, and generating synthetic data through the reverse diffusion process.

The key learning strategy is the co-evolving conditional diffusion process. During training, the continuous diffusion model uses the noisy categorical variables as its condition, while the discrete diffusion model uses the noisy continuous variables as its condition. This allows the two models to learn the relationships between continuous and categorical features rather than modelling them independently.

CoDi also uses contrastive learning with positive and negative conditions to strengthen the dependency between the two variable types and improve the coherence of the generated synthetic records.

### Training Details
CoDi jointly trains a continuous diffusion model and a discrete diffusion model. At each training step, a diffusion timestep is randomly sampled and noise is added to both continuous and categorical variables according to their corresponding diffusion processes.

The continuous diffusion model learns to predict the Gaussian noise added to the continuous variables while using the noisy categorical variables as its condition. The discrete diffusion model learns the reverse categorical diffusion process while using the noisy continuous variables as its condition.

The two models are therefore trained together in a co-evolving process rather than independently. CoDi also introduces contrastive learning using positive and negative conditions to strengthen the learned relationships between continuous and categorical variables. The continuous and discrete models are optimized separately using their corresponding diffusion and contrastive losses.

### Convergence Criteria
The current CoDi implementation uses a fixed number of training epochs as its stopping criterion. Training stops once the specified `epochs` value has been reached.

No additional early-stopping criterion or loss threshold is currently used. The number of epochs can be configured as a model hyperparameter depending on the experimental setup.

---

## Hyperparameters
CoDi uses separate hyperparameters for the continuous and discrete diffusion models. The research paper specifies both shared diffusion parameters and dataset-specific training parameters.

### Paper-Specified Hyperparameters

| Hyperparameter | Value / Setting | Description |
|---|---|---|
| Diffusion timesteps (`T`) | `50` | Number of forward and reverse diffusion steps |
| Beta schedule | Linear | Noise variance schedule |
| Starting beta (`beta_1`) | `1e-5` | Initial noise level |
| Final beta (`beta_T`) | `2e-2` | Final noise level |
| Optimizer | Adam | Optimizer used to train the diffusion models |
| Batch size | `512` | Number of samples processed in each training batch |
| Continuous learning rate | Dataset-specific | Learning rate for the continuous diffusion model |
| Discrete learning rate | Dataset-specific | Learning rate for the discrete diffusion model |
| Continuous embedding dimension | Dataset-specific | Embedding dimension used by the continuous model |
| Discrete embedding dimension | Dataset-specific | Embedding dimension used by the discrete model |
| Continuous hidden dimensions | Dataset-specific | Hidden-layer configuration of the continuous model |
| Discrete hidden dimensions | Dataset-specific | Hidden-layer configuration of the discrete model |
| `lambda_con` | Dataset-specific | Weight of the continuous contrastive loss |
| `lambda_dis` | Dataset-specific | Weight of the discrete contrastive loss |

### Current Katabatic Parameters

The current implementation exposes the major CoDi parameters through the model constructor, allowing the paper-specific settings to be reproduced for individual datasets.

During the earlier Model Validation stage, `epochs`, `batch_size`, and learning rate were experimentally varied to investigate model performance and efficiency. These tuning experiments are not treated as part of the final paper-reproduction configuration. Following the updated project objective, the model should use the hyperparameters reported in the original CoDi paper wherever they are specified.

---

## Input
CoDi accepts standard tabular data containing both feature variables and an optional target label.

- `X`: Tabular feature matrix containing continuous and/or categorical features.
- `y`: Target labels associated with the input samples.

When `y` is provided, it is combined with `X` during training so that the target label is synthesised together with the feature data.

### Expected Files
The CoDi implementation follows the standard Katabatic training data format:

- `x_train.csv`: Contains the training feature data.
- `y_train.csv`: Contains the corresponding target labels.

If `y_train.csv` is provided, the target column is combined with `x_train.csv` during CoDi training so that both features and labels can be synthesised together.

---

## Preprocessing

CoDi performs additional preprocessing after the standard Katabatic data preparation pipeline. The model separates the input data into numerical and categorical features because the two feature types are handled by different diffusion processes.

### Numerical Features

Numerical features are treated as continuous variables. CoDi applies Min-Max scaling to transform each continuous feature into the range `[-1, 1]` before training.

The minimum and maximum values of each numerical feature are stored in the model schema. After synthetic data generation, these values are used to transform the generated numerical features back to their original scale.

### Categorical Features

Categorical features are first encoded into integer category indices. The number of categories for each feature is recorded in the model schema.

Before being processed by the discrete diffusion model, categorical features are converted into one-hot representations. The one-hot representation is also used when categorical variables are provided as the condition for the continuous diffusion model.

During synthetic data generation, the generated categorical representations are converted back into category indices and then decoded to their original categorical values.


## Label Handling

When target labels are provided in `y_train.csv`, CoDi combines the labels with the feature data before training. This allows the target variable to be modelled and synthesised together with the other columns rather than being used only as an external condition.

After synthetic data generation, the target column is separated from the generated feature data. The synthetic features are saved to `x_synth.csv`, while the generated target labels are saved to `y_synth.csv`.

---

## Output

After training and synthetic data generation, CoDi produces the following output files:

- `x_synth.csv`: Contains the generated synthetic feature data.
- `y_synth.csv`: Contains the generated synthetic target labels.
- `metadata.json`: Contains metadata associated with the model training and synthetic data generation process.

The generated synthetic data preserves the original feature structure and column order so that it can be directly used by the downstream Katabatic TSTR evaluation pipeline.

---

## Evaluation

The `evaluate()` method evaluates the quality of the generated synthetic data using the TSTR (Train on Synthetic, Test on Real) approach.

Downstream machine learning models are trained using the synthetic dataset and then evaluated on the real test dataset. The method returns the evaluation results as a dictionary containing the performance metrics for each downstream model.

The current Katabatic evaluation pipeline uses models such as Logistic Regression (LR), Multi-Layer Perceptron (MLP), Random Forest (RF), and XGBoost. For classification datasets, the reported metrics include:

- Accuracy
- F1 Score

The evaluation results are also stored as a TSTR report for subsequent benchmarking and comparison.

---

## Strengths
- CoDi is specifically designed for mixed-type tabular data containing both continuous and categorical variables.
- It uses separate Gaussian and categorical diffusion processes, allowing different feature types to be modelled in their appropriate representation spaces.
- The co-evolving conditional diffusion approach helps preserve relationships between continuous and categorical variables.
- CoDi can be applied to datasets with different feature compositions, including numerical, categorical, and mixed-type tabular data.
- The model can achieve strong downstream predictive performance by generating synthetic data that preserves useful information from the original dataset.
- The relatively small number of diffusion steps used by CoDi helps reduce sampling cost compared with diffusion models that require a much larger number of steps.

---

## Limitations

- CoDi requires two separate diffusion models for continuous and categorical variables, which increases training complexity and computational cost.
- The model can require more training and sampling time than simpler tabular synthesis methods because both continuous and discrete diffusion processes must be performed.
- Model performance can be sensitive to hyperparameter settings, and the original paper uses different configurations for different datasets.
- CoDi is primarily designed for mixed-type tabular data, so its advantages may be less significant for datasets containing only a single feature type.
- The co-evolving and contrastive learning mechanisms make the model more complex to implement and maintain compared with simpler generative models.
- As a diffusion-based model, synthetic data generation requires multiple reverse diffusion steps, which can increase inference time when generating large numbers of samples.

---
## Installation
```bash
poetry install --extras codi
```
--- 

## Usage

Evaluation pipeline benchmark scripts for each dataset:

- Adult: `benchmarks/examples/codi/`
- Car: `benchmarks/examples/codi/`
- Magic: `benchmarks/examples/codi/`
- Nursery: `benchmarks/examples/codi/`
- Shuttle: `benchmarks/examples/codi/`

```python
from katabatic.models.codi import CODI

model = CODI(
    n_steps=50,
    beta_1=1e-5,
    beta_T=0.02,
    epochs=30,
    batch_size=512,
    lr_con=2e-3,
    lr_dis=2e-3,
    lambda_con=0.2,
    lambda_dis=0.2,
    random_state=42
)

model.train(
    dataset_dir="path_to_data",
    synthetic_dir="path_to_save"
)

synthetic_data = model.sample(1000)
```

The hyperparameters can be adjusted according to the dataset-specific configuration required for paper reproduction.

--- 

## Model Evaluation Benchmarks Results

## Model Evaluation Benchmarks Results

#### Adult Dataset

TSTR Evaluation Results:

- LR
  - Accuracy: 0.2670
  - F1 Score: 0.1634
  - AUC: 0.6983

- MLP
  - Accuracy: 0.2564
  - F1 Score: 0.1320
  - AUC: 0.5046

- RF
  - Accuracy: 0.7976
  - F1 Score: 0.7514
  - AUC: 0.8133

- XGBoost
  - Accuracy: 0.8085
  - F1 Score: 0.7814
  - AUC: 0.8301


#### Car Dataset

TSTR Evaluation Results:

- LR
  - Accuracy: 0.6965
  - F1 Score: 0.5794

- MLP
  - Accuracy: 0.6908
  - F1 Score: 0.5774

- RF
  - Accuracy: 0.6994
  - F1 Score: 0.5818

- XGBoost
  - Accuracy: 0.6850
  - F1 Score: 0.5781


#### Magic Dataset

TSTR Evaluation Results:

- LR
  - Accuracy: 0.7158
  - F1 Score: 0.6987
  - AUC: 0.7019

- MLP
  - Accuracy: 0.7269
  - F1 Score: 0.7019
  - AUC: 0.6935

- RF
  - Accuracy: 0.7516
  - F1 Score: 0.7290
  - AUC: 0.7617

- XGBoost
  - Accuracy: 0.7240
  - F1 Score: 0.7105
  - AUC: 0.7423


#### Nursery Dataset

TSTR Evaluation Results:

- LR
  - Accuracy: 0.5525
  - F1 Score: 0.5399

- MLP
  - Accuracy: 0.7662
  - F1 Score: 0.7602

- RF
  - Accuracy: 0.8086
  - F1 Score: 0.8016

- XGBoost
  - Accuracy: 0.8040
  - F1 Score: 0.7977


#### Shuttle Dataset

TSTR Evaluation Results:

- LR
  - Accuracy: 0.9213
  - F1 Score: 0.9163

- MLP
  - Accuracy: 0.6771
  - F1 Score: 0.7030

- RF
  - Accuracy: 0.9695
  - F1 Score: 0.9676

- XGBoost
  - Accuracy: 0.9036
  - F1 Score: 0.9083

#### Comparison with the CoDi Paper

The current Katabatic evaluation follows the same general TSTR principle
used to evaluate the utility of synthetic data. However, some differences
were identified between the current evaluation implementation and the
evaluation protocol reported in the original CoDi paper.

In particular, the current Katabatic TSTR implementation calculates
multi-class F1 using the weighted F1 score, whereas the CoDi paper
reports Macro F1 for multi-class classification. Therefore, the F1
scores reported above should not be interpreted as a direct numerical
reproduction of the paper's multi-class F1 results.

Another difference is AUROC evaluation. The current Katabatic TSTR
implementation reports AUC only for binary classification tasks. As a
result, AUC is available for the Adult and Magic datasets but is not
reported for the multi-class Car, Nursery, and Shuttle datasets. The CoDi
paper, in contrast, also reports AUROC for multi-class classification.

The datasets currently used by the Katabatic benchmarking pipeline are
also not identical to all datasets used in the original CoDi experiments.
Therefore, the current results are primarily used to validate the behaviour
and downstream utility of the Katabatic CoDi implementation rather than as
a direct dataset-level reproduction of the paper's reported scores.

A paper-aligned evaluation would require using Macro F1 for multi-class
classification and extending the current evaluation to support multi-class
AUROC.
  

## Model Performance Benchmarks Results

The following table reports the runtime of the CoDi model on the five
benchmark datasets. All experiments were executed on CPU using the same
baseline configuration.

| Dataset | Training Time (s) | Inference Time (s) | Total Pipeline Time (s) |
|---|---:|---:|---:|
| Adult | 42.68 | 3.88 | 51.17 |
| Car | 1.26 | 0.22 | 2.04 |
| Magic | 9.72 | 1.42 | 13.25 |
| Nursery | 6.02 | 0.87 | 9.21 |
| Shuttle | 29.14 | 3.96 | 37.79 |

The runtime varies depending on the size and structure of each dataset.
Among the five datasets, Car required the shortest total pipeline time,
while Adult required the longest. These results provide a baseline for
comparing the computational performance of the CoDi implementation across
different datasets.