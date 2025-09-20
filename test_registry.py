# test_registry.py

from Models import MODEL_REGISTRY
from sklearn.datasets import load_iris
import os

# Initialize your PATEGAN model
model = MODEL_REGISTRY["Vidushi_PATEGAN"]()

# Load a small dataset (Iris)
X, y = load_iris(as_frame=True, return_X_y=True)
X["target"] = y

# Train and generate
model.train(X, y)
synth = model.generate(5)

print(" Registry + Wrapper test successful!")
print(synth)

# Save results
output_path = "experiments/pategan/iris_synth_registry.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
synth.to_csv(output_path, index=False)
print(f" Synthetic data saved to: {output_path}")
