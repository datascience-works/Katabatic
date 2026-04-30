
# CTGAN Model (Custom Implementation)

## Model Overview
This is a custom PyTorch-based implementation of CTGAN that avoids external dependencies such as SDV.

## Approach
The model uses a GAN architecture with a generator and discriminator trained adversarially.
Basic preprocessing is applied to handle categorical and missing data.

## Training Details
- Epoch-based training
- epochs = 50
- batch_size = 128

## Hyperparameters
- epochs = 50
- batch_size = 128
- learning_rate = 0.001
- noise_dim = 32

## Input
- Tabular dataset (DataFrame or numpy array)

## Output
- Synthetic tabular data

## Usage
from models import CTGANModel

model = CTGANModel()
model.fit(X_train)

synthetic = model.sample(1000)

## Notes
- No external CTGAN libraries used
- Simplified implementation
- Does not include conditional sampling

## Datasets Tested
- Adult
- Car
- Magic
- Nursery
- Shuttle

## Runtime
- Moderate depending on dataset size
