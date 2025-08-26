# Tabddpm - TabDDPM: Modelling Tabular Data with Diffusion Models

## Setup environment
1. install python ( Prefer 3.9 and latest)
2. clone the repo, enter the folder and run the requiremnts
```bash

#clone the folder
git clone --branch feature/TabDDPM https://github.com/datascience-works/Katebatic.git

#enter the katabatic folder
cd Katebatic


#installing requirements
pip install -r requiremnts.txt

```

## Model wrapping file - trial_tddpm
1. Implemented based on Sklearn structure
```bash

from katebatic.models.TabDDPM import tabddpm
```

2. Environment Check
```bash
import os
os.getcwd()
```

3. Dataset Fetching (UCI Repository)
```bash

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = abalone.data.features 
y = abalone.data.targets 

df = pd.concat([X,y], axis=1)
```

4. Model structure
- steps → Indicates the total number of training steps for the diffusion process, which influences the duration of model training.
- lr (Learning Rate) → Refers to the step size utilized in gradient descent optimization.
- batch_size → Represents the number of samples that are processed prior to updating the weights; a larger batch size results in quicker training but increases memory consumption.
- model_type → Specifies the selection of the backbone model, such as "mlp" for multilayer perceptron or "transformer" if it is supported.
- d_layers → Determines the hidden layers within the neural network, for instance, [256, 256] signifies two hidden layers, each containing 256 units.
- dropout → Indicates the dropout probability used for regularization, where 0.0 signifies no dropout.
- num_classes → Relevant in classification tasks; a value of 0 indicates regression, while a value greater than 0 specifies the number of target classes.
- is_y_cond → A flag for conditional training, where True indicates conditioning on labels y and False indicates unconditional training.
- normalization → Refers to the method employed for input preprocessing, such as "quantile" or "standard".
- device → Specifies whether to use "cpu" or "cuda" (the latter if a GPU is available).
- seed → The random seed utilized to ensure reproducibility.

```bash
# Initialize and train
model = tabddpm.TabDDPM(
    steps=1000,
    lr=0.001,
    weight_decay=1e-5,
    batch_size=4096, 
    model_type='mlp', 
    d_layers=[256, 256],
    dropout=0.0,
    num_classes=0,
    is_y_cond=False,
    normalization='quantile',
    device='cpu',
    seed=1
)

model.fit(X, y)

# Generate synthetic data
# X_syn, y_syn = model.generate(n_samples=100)
synthetic_df = model.generate_df(n_samples=150)
```


```