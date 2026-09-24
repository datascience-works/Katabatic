# GANBLRPP

GANBLRPP extends GANBLR by adding support for numerical columns.

## Overview

The model inherits from the existing GANBLR implementation and uses `DMMDiscretizer` to handle numerical columns before training. After synthetic data is generated, the numerical columns are converted back to their original numerical form.

## Usage

```python
from katabatic.models.ganblrpp import GANBLRPP

model = GANBLRPP(
    numerical_columns=["column_name"]
)

model.fit(X, y)

synthetic_data = model.sample(size=100)
## References

- Zhang, Y., Zaidi, N. A., Zhou, J., & Li, G. (2022). *GANBLR++: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks.* SDM 2022. doi:10.1137/1.9781611977172.34
