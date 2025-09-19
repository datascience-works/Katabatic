# cWGAN-based implementation

Implementation of the cWGAN-based oversampling method. 
Fits a conditional Wasserstein GAN with Gradient Penalty 
and an auxiliary classifier loss to a tabular dataset with categorical and numerical attributes.
The fitted cWGAN model can than be used to resample an imbalanced training set. 
Currently only supports binary classification.

## Implementation
This implementation is based on 
https://github.com/justinengelmann/GANbasedOversampling.git

## New Feature Added (Rajat Dulal):
1. Functionality extended to generate synthetic dataset for both classes rather than for minority class only for dataset balancing.

2. Synthetic data now is able to output in the form of dataframe and numpy array both as desired.

## Model Configuration

### Core Parameters

```python
gan = WGANGP(
    write_to_disk=True,              # Creates output folder for results
    compute_metrics_every=1250,      # Frequency of metric computation
    print_every=2500,                # Training progress print frequency
    plot_every=10000,                # Plotting frequency
    num_cols=num_cols,               # Number of numerical columns
    cat_dims=cat_dims,               # Categorical column dimensions
    transformer=prep.named_transformers_['cat']['onehotencoder'],
    cat_cols=cat_cols,               # Categorical column names
    use_aux_classifier_loss=True,    # Enable auxiliary classifier
    d_updates_per_g=3,               # Discriminator updates per generator update
    gp_weight=15                     # Gradient penalty weight
)
```

### Training Configuration

```python
gan.fit(
    X_train_trans, 
    y=y_train.values,
    condition=True,                  # Enable conditional generation
    epochs=300,                      # Training epochs
    batch_size=64,                   # Batch size
    netG_kwargs={...},              # Generator architecture
    netD_kwargs={...}               # Discriminator architecture
)
```

## Usage Example

### Basic Training
```python
# Initialize the GAN
gan = WGANGP(write_to_disk=True, ...)

# Train the model
gan.fit(X_train_trans, y=y_train.values, condition=True, epochs=300, ...)
```

### Synthetic Data Generation
```python
# Generate 100 synthetic samples with custom class distribution
dataset_new = gan.generate_df(n=100, sampling_rate={25:75})
```

The `sampling_rate` parameter allows you to specify the desired distribution of classes in the generated data (e.g., 25% class 0, 75% class 1).

## Author
Rajat Dulal

rjtdulal@gmail.com

Deakin University