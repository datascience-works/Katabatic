from tabebm.TabEBM import TabEBM
from sklearn.datasets import load_iris
import numpy as np

# Use a real small dataset, similar in spirit to the paper's small-sample regime
X, y = load_iris(return_X_y=True)

# Simulate a small-sample scenario (paper tests Nreal in [20, 50, 100, 200, 500])
rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=30, replace=False)
X_small, y_small = X[idx], y[idx]

print(f"Training on {len(X_small)} real samples across {len(np.unique(y_small))} classes")

tabebm = TabEBM()
synthetic = tabebm.generate(X_small, y_small, num_samples=50)

for class_id, data in synthetic.items():
    print(f"{class_id}: generated {data.shape[0]} samples, {data.shape[1]} features")
    print(f"  mean: {data.mean(axis=0).round(2)}")
    print(f"  real class mean: {X_small[y_small == int(class_id.split('_')[-1])].mean(axis=0).round(2)}")
