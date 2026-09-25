# GMM

GMM - Gaussian Mixture Model is a probabilistic model that represents data as a combination of multiple Gaussian distributions.

For synthetic tabular data generation, the model learns patterns from the real dataset and generates new samples from the learned Gaussian mixture distributions.

This Katabatic implementation uses a **class-conditional GMM approach**, where a separate Gaussian Mixture Model is trained for each target class. This helps preserve the target distribution while generating synthetic feature values.

***

## Overview

This implementation integrates the following core ideas, **adapted specifically for Katabatic**:

- **Class-conditional modelling**: A separate Gaussian Mixture Model is fitted for each target class.
- **Mixed data support**: Categorical features are encoded as integer values before model fitting and converted back to their original labels after synthetic data generation.
- **Feature scaling**: Features are standardized before fitting the Gaussian Mixture Models.
- **Target distribution preservation**: Synthetic samples are generated according to the class distribution observed in the real training data.
- **Reproducible sampling**: Random seeds can be supplied during generation to support Katabatic stability evaluation.

***

## Implementation Details

**Algorithm**: Gaussian Mixture Model using Expectation-Maximization.

**Library**: The implementation uses `GaussianMixture` and `StandardScaler` from scikit-learn.

**Source Implementation**: Adapted from the GMM implementation contributed by **Rishi Goyal** to the Katabatic mentorship repository.

**Repository branch**:
https://github.com/katabatic-mentorship/katabatic-mentorship-repo/tree/Rishi_Goyal

**Original model location**:
`katabatic/models/gmm_Rishi`

### Original / Reference Implementation Recipe

```text
GMMModel(
    target_col,
    n_components=4,
    covariance_type="full",
    random_state=42
)

fit(df)

For each target class:
    Detect feature types
    Encode categorical features
    Standardize feature values
    Fit a GaussianMixture model

generate(n_rows)
```

### Katabatic Implementation

**Training Process**:

```text
1. Identify the target and feature columns.
2. Detect continuous and categorical feature types.
3. Encode categorical features as integer values.
4. Calculate the target-class distribution.
5. Separate the training data by target class.
6. Standardize the feature values for each class.
7. Fit one Gaussian Mixture Model for each target class.
```

**Synthetic Data Generation**:

```text
1. Calculate how many synthetic rows are required for each target class.
2. Sample synthetic feature values from each class-specific GMM.
3. Reverse the feature scaling.
4. Round features that were originally integers.
5. Convert encoded categorical values back to their original labels.
6. Restore the target column.
7. Return the requested number of synthetic rows.
```

For compatibility with the Katabatic evaluation pipeline, the model also provides:

```python
sample(n_rows, seed=None)
```

This allows reproducible sampling during stability evaluation.

***

## Hyperparameter Comparison

| Parameter | Original Implementation | Katabatic |
|---|---|---|
| n_components | 4 | **4** |
| covariance_type | full | **full** |
| random_state | 42 | **42** |
| target_col | Required | **Required** |
| Sampling seed | Constructor seed | **Optional seed supported during sampling** |

Unlike neural-network-based generative models, GMM does not require epochs or batch sizes.

***

## Data Processing

**Categorical features**:
- Categorical columns are detected from object or categorical data types.
- Categories are converted into integer values before model fitting.
- Generated categorical values are rounded and clipped to valid category indices.
- Integer codes are converted back to the original category labels.

**Continuous features**:
- Continuous features are converted to numeric values.
- Features are standardized using `StandardScaler`.
- Generated values are transformed back to their original scale.
- Columns that were originally integers are rounded back to integer values.

**Target column**:
- A separate GMM is trained for each target class.
- Synthetic class counts follow the target-class proportions in the training data.

***

## Dependencies

The GMM implementation uses:

- NumPy
- pandas
- scikit-learn

The model does not require an external account, API key, external service, or proprietary licence to operate.

***

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*. Journal of the Royal Statistical Society: Series B, 39(1), 1-38.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- scikit-learn Gaussian Mixture documentation: https://scikit-learn.org/stable/modules/mixture.html
- Source implementation by Rishi Goyal: https://github.com/katabatic-mentorship/katabatic-mentorship-repo/tree/Rishi_Goyal

## Generative AI Acknowledgement

AI was used to assist with interpreting and structuring the GMM implementation based on the existing source implementation, established Gaussian Mixture Model methodology, and Katabatic model requirements.

All generated content was manually verified, modified, and extended, including code restructuring, Katabatic pipeline integration, debugging, documentation, and experimental validation. The final implementation reflects the author's independent work.
