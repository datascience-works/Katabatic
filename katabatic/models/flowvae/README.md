# Flow-VAE Model

## Model Overview
Flow-VAE (Flow-based Variational Autoencoder) is a **deep generative model** for tabular data that combines:

- **Variational Autoencoders (VAE)** for latent representation learning  
- **Normalizing Flows** for improving the flexibility of the latent distribution  

The model learns a **latent space representation** of tabular data and generates synthetic samples by decoding transformed latent vectors sampled from a learned probabilistic distribution.

This implementation is adapted into the **Katabatic 3-file format** and is fully self-contained using PyTorch. It is based on the uploaded Flow-VAE architectures and converted for tabular CSV generation.

---

## Approach
The model follows a VAE + Flow pipeline:

- Encodes tabular data into a latent Gaussian space using an **encoder network**
- Learns latent distribution parameters:
  - Mean (μ)
  - Log variance (logσ²)
- Samples latent vectors using the **reparameterization trick**
- Applies **normalizing flow transformations**
- Decodes transformed latent vectors into synthetic tabular samples

### Key Idea
A standard VAE assumes:

```math
q(z|x)=\mathcal{N}(\mu,\sigma^2 I)
```

which can be too restrictive for complex datasets.

Flow-VAE improves this by transforming latent variables through a sequence of invertible flow functions:

```math
z_k=f_k\circ f_{k-1}\circ ... \circ f_1(z_0)
```

This creates a richer latent distribution capable of modeling more complicated data structures.

---

## Flow Types Supported
The implementation supports multiple normalizing flow variants:

- **Planar Flow**
- **Radial Flow**
- **Householder Flow**
- **NICE Flow**

Each flow incrementally transforms the latent space while maintaining tractable likelihood estimation.

---

## Training Details
- Neural network training using **PyTorch**
- Mini-batch gradient descent with Adam optimizer

### Loss Components

#### Reconstruction Loss
Measures how closely reconstructed data matches the original input:

```math
\mathcal{L}_{recon}=||x-\hat{x}||^2
```

#### KL Divergence Loss
Regularizes the latent distribution toward a Gaussian prior:

```math
D_{KL}(q(z|x)\parallel p(z))
```

#### Flow Jacobian Correction
Normalizing flows require a Jacobian determinant correction:

```math
\log\left|\det\frac{\partial f}{\partial z}\right|
```

### Total Loss

```math
\mathcal{L}=\mathcal{L}_{recon}+D_{KL}-\log|\det J|
```

---

## Training Loop
- Encode batch → obtain μ and logσ²  
- Sample latent vector z₀  
- Apply flow transformations z₀ → zₖ  
- Decode zₖ → reconstructed samples  
- Compute total loss  
- Backpropagate and update parameters  

---

## Hyperparameters
Defined in `FlowVAEModel`:

- `epochs = 50`
- `batch_size = 128`
- `hidden_dim = 128`
- `latent_dim = 16`

- `layers = 2`
- `gate = False`

- `flow_type = "planar"`
- `flow_length = 2`

- `learning_rate = 1e-3`
- `random_state = 42`

- `device = cuda / cpu`

---

## Input
- `X`: Tabular dataset
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

The model combines features and labels into a **single encoded training matrix** before learning the joint distribution.

---

## Preprocessing
The implementation automatically handles mixed tabular data types:

### Numerical Features
- Z-score normalization:

```math
x' = \frac{x-\mu}{\sigma}
```

### Categorical Features
- One-hot encoding

### During Generation
- Numerical values are inverse-scaled
- Categorical values are reconstructed using argmax decoding

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

The synthetic data attempts to preserve:

- Feature distributions
- Feature correlations
- Latent structure
- Target-feature relationships

Synthetic outputs are automatically separated back into X/y format compatible with Katabatic evaluation pipelines.

---

## Important Notes

### Label Handling
The Flow-VAE generates continuous outputs, which can create invalid categorical labels for classification tasks.

To avoid this:
- Synthetic labels are sampled from the original training label distribution
- Labels are remapped into consecutive integer classes when necessary for compatibility with XGBoost

---

### Strengths
- More expressive latent distribution than standard VAE
- Better modeling of multimodal distributions
- Flexible latent geometry via flows
- Fully self-contained PyTorch implementation

---

### Limitations
⚠️ The model may struggle with:
- Highly imbalanced datasets
- Very small datasets
- High-cardinality categorical features

⚠️ Uses:
- MSE reconstruction loss for all features
- Argmax decoding for categorical variables

These choices may not perfectly model complex categorical probability distributions.

---

## Usage

```python
from models import FlowVAEModel

model = FlowVAEModel()

model.train(
    data_dir="path_to_data",
    synthetic_dir="path_to_save"
)

df_synth = model.sample(1000)
```