# Katabatic — PATE-GAN Integration

This README documents the end-to-end implementation of **Vidushi_PATEGAN** in the Katabatic framework, including environment setup, wrapper integration, UI routes, testing, and experiment logging.

---

## 1. Environment Setup

1. Clone the Katabatic repository:
   ```bash
   git clone https://github.com/datascience-works/Katebatic.git
   cd Katebatic


Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows PowerShell


Install required dependencies:

pip install -r requirements.txt
pip install synthcity flask flask-cors scikit-learn

2. Model Wrapper (Models/Vidushi_PATEGAN/__init__.py)
import os
from synthcity.plugins import Plugins
from sklearn.datasets import load_iris
from Models import register_model

@register_model("Vidushi_PATEGAN")
class VidushiPATEGAN:
    def __init__(self):
        self.plugin = Plugins().get(
            "pategan",
            n_iter=5,
            n_teachers=3,
            teacher_template="linear",
            random_state=42,
        )

    def train(self, X, y=None):
        X["target"] = y
        self.plugin.fit(X)
        return self

    def generate(self, n=10):
        return self.plugin.generate(n).dataframe()

    def save_results(self, output_path="experiments/pategan/iris_synth.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.generate(150).to_csv(output_path, index=False)
        print(f"Saved synthetic data -> {output_path}")

if __name__ == "__main__":
    X, y = load_iris(as_frame=True, return_X_y=True)
    model = VidushiPATEGAN().train(X, y)
    synth = model.generate(5)
    print(synth.head())
    model.save_results()

🗂 3. Register Model

Update Models/__init__.py to import and register the new wrapper:

from .Vidushi_PATEGAN import VidushiPATEGAN


This ensures MODEL_REGISTRY["Vidushi_PATEGAN"] works globally.

🖥 4. Testing the Registry

Create test_registry.py in the root of the project:

from Models import MODEL_REGISTRY
from sklearn.datasets import load_iris

# Initialize model
model = MODEL_REGISTRY["Vidushi_PATEGAN"]()

# Load dataset
X, y = load_iris(as_frame=True, return_X_y=True)
X["target"] = y

# Train & generate
model.train(X, y)
synth = model.generate(5)

print("Registry + Wrapper test successful!")
print(synth.head())

# Save synthetic data
model.save_results("experiments/pategan/iris_synth_registry.csv")


Run it:

python test_registry.py

5. UI Integration (Flask)

Add UIKatabatic/app.py:

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from flask import Flask, jsonify
from Models import MODEL_REGISTRY
from sklearn.datasets import load_iris

app = Flask(__name__)

@app.route("/")
def home():
    return "Katabatic UI is running!"

@app.route("/pategan/generate")
def generate_pategan():
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    model = MODEL_REGISTRY["Vidushi_PATEGAN"]().train(X, y)
    synth = model.generate(5).to_dict(orient="records")
    return jsonify(synth)

@app.route("/pategan/save")
def save_pategan():
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    model = MODEL_REGISTRY["Vidushi_PATEGAN"]().train(X, y)
    output_path = "experiments/pategan/iris_synth_ui.csv"
    model.save_results(output_path)
    return jsonify({"message": f"Synthetic data saved to {output_path}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)


Run Flask:

cd UI/UIKatabatic
python app.py


Endpoints:

http://127.0.0.1:5000/ → UI check

http://127.0.0.1:5000/pategan/generate → returns synthetic samples

http://127.0.0.1:5000/pategan/save → saves CSV in experiments/pategan/

6. Experiment Logging

Create experiments/pategan/<date>.md:

# PATE-GAN — Iris (UI run)
- **Model:** Vidushi_PATEGAN
- **Dataset:** sklearn Iris (150 rows, 4 features + target)
- **Params:** n_iter=5, n_teachers=3, teacher_template="linear", random_state=42
- **Generated:** 150 rows
- **Artifacts**
  - CSV: `experiments/pategan/iris_synth_ui.csv`
- **Notes:** Smoke test via `/pategan/generate` and `/pategan/save` succeeded.

7. Workflow Summary

 Implemented Vidushi_PATEGAN wrapper.

 Registered in Models/init.py.

 Verified with test_registry.py.

 Added Flask UI routes (/generate, /save).

 Logs written to experiments/pategan/.




