# TVAEGAN Model

## Model Overview
TVAEGAN (Tabular Variational Autoencoder Generative Adversarial Network) is a **hybrid deep generative model** for tabular data that combines:

- **Variational Autoencoders (VAE)** for latent representation learning  
- **Generative Adversarial Networks (GAN)** for realistic data generation  

The model learns a **latent space representation** of tabular data and uses adversarial training to improve the realism of generated samples.

---

## Approach
The model follows a hybrid VAE-GAN pipeline:

- Encodes tabular data into a latent space using a **VAE-style encoder**
- Samples latent vectors from a **Gaussian prior**
- Decodes latent vectors into synthetic data using a **generator network**
- Uses a **critic (discriminator)** to distinguish real vs synthetic samples
- Optimizes:
  - Reconstruction loss (data fidelity)
  - Regularization loss (latent distribution alignment)
  - Adversarial loss (realism)

**Key idea:**
- Encoder learns structured latent representation  
- Generator produces realistic tabular samples  
- Critic enforces distributional similarity  

---

## Training Details
- Neural network-based training using **PyTorch**
- Alternating optimization between:
  - Encoder (VAE objective)
  - Generator (adversarial objective)
  - Critic (Wasserstein-style loss)

### Loss Components
- **Reconstruction loss (MSE)**  
- **Energy distance loss** for latent regularization  
- **Adversarial loss** via critic  

### Training Loop
- Encoder step → improves reconstruction + latent structure  
- Generator step → fools critic  
- Critic step → distinguishes real vs fake  

---

## Hyperparameters
Defined in `TVAEGANSynthesizer`:

- `epochs = 700`  
- `batch_size = 500`  
- `latent_dim = number_of_columns`  

- `cat_emb_size = 25`  
- `num_emb_size = 25`  

- `w_regularize = 1`  
- `w_reconstruct = 10`  

- `lr_encoder = 5e-5`  
- `lr_critic = 5e-5`  
- `lr_generat = 5e-5`  

- `s_generat = 5`  
- `s_encoder = 5`  

- `clip = 0.01`  
- `dropout = 0.1`  
- `random_state = 42`  

---

## Input
- `X`: Tabular dataset  
- `y`: Target labels  

**Expected files:**
- `x_train.csv`  
- `y_train.csv`  

The model combines `X` and `y` into a **single training table** and learns their joint distribution.

---

## Output
- `x_synth.csv`: Synthetic features  
- `y_synth.csv`: Synthetic labels  
- `synthetic.csv`: Full synthetic dataset  

Generated samples preserve:
- Feature distributions  
- Relationships between variables  
- Approximate class structure  

---

## Important Notes
- The model **jointly generates features and labels**
- Synthetic labels may:
  - Be non-contiguous (e.g. `0, 3, 4`)  
  - Miss classes in small samples  

⚠️ Because of this:
- Downstream models (e.g. XGBoost) may require **label encoding** during evaluation  

---

## Usage
```python
from models import TVAEGANModel

model = TVAEGANModel()

model.train(
    dataset_dir="path_to_data",
    synthetic_dir="path_to_save"
)

df_synth = model.sample(1000)