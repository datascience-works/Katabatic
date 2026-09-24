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
