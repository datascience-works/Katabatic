# TVAE Model

## Model Overview
TVAE (Tabular Variational Autoencoder) is a **deep generative model** for tabular data based on:

- **Variational Autoencoders (VAE)** for probabilistic representation learning  

The model learns a **latent space representation** of tabular data and generates synthetic samples by decoding latent vectors sampled from a Gaussian distribution.

This implementation is **fully self-contained** and removes external dependencies such as SDV, making it lightweight and compatible with Katabatic.

---

## Approach
The model follows a standard VAE pipeline:

- Encodes tabular data into a latent space using an **encoder network**
- Learns parameters of a **Gaussian latent distribution** (mean and variance)
- Samples latent vectors using the **reparameterization trick**
- Decodes latent vectors into synthetic data using a **decoder network**

**Key idea:**
- Encoder compresses data into latent representation  
- Latent space captures underlying data structure  
- Decoder reconstructs realistic tabular samples  

---

## Training Details
- Neural network-based training using **PyTorch**
- Mini-batch gradient descent with Adam optimizer  

### Loss Components
- **Reconstruction loss (MSE)** — ensures generated data matches original data  
- **KL divergence loss** — regularizes latent space toward Gaussian prior  

### Training Loop
- Encode batch → obtain μ and logσ²  
- Sample latent vector z  
- Decode z → reconstructed data  
- Compute loss = reconstruction + β × KL  
- Backpropagate and update parameters  

---

## Hyperparameters
Defined in `TVAEModel`:

- `epochs = 100`  
- `batch_size = 256`  
- `latent_dim = 32`  
- `hidden_dim = 128`  

- `learning_rate = 1e-3`  
- `beta = 0.01`  

- `sample_size = None`  
- `random_state = 42`  

- `device = cuda / cpu`  

---

## Input
- `X`: Tabular dataset  
- `y`: Target labels  

**Expected files:**
- `x_train.csv`  
- `y_train.csv`  

The model combines `X` and `y` into a **single training table** before learning the joint distribution.

---

## Output
- `x_synth.csv`: Synthetic features  
- `y_synth.csv`: Synthetic labels  

Generated samples aim to preserve:
- Feature distributions  
- Relationships between variables  
- Target-feature dependencies  

Synthetic data is automatically split back into X and y format.

---

## Important Notes
- The model **jointly models features and target**
- Uses:
  - Z-score normalization for numerical features  
  - One-hot encoding for categorical features  

- During generation:
  - Categorical values are reconstructed using **argmax**
  - Numerical values are **inverse-scaled**

⚠️ Limitations:
- Uses MSE loss for all features (not ideal for categorical distributions)  
- May struggle with:
  - Highly imbalanced datasets  
  - Complex multimodal relationships  

---

## Usage
```python
from models import TVAEModel

model = TVAEModel()

model.train(
    data_dir="path_to_data",
    synthetic_dir="path_to_save"
)

df_synth = model.sample(1000)