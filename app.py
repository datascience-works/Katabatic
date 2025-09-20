import sys
import os

# Ensure project root (Katabatic/) is on the Python path
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
    # Load a small dataset
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y

    # Create and train the registered model (Vidushi_PATEGAN)
    model = MODEL_REGISTRY["Vidushi_PATEGAN"]().train(X, y)

    # Generate a few samples
    synth = model.generate(5).to_dict(orient="records")
    return jsonify(synth)

#  Paste this new route here
@app.route("/pategan/save")
def save_pategan():
    # Load dataset
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y

    # Train model
    model = MODEL_REGISTRY["Vidushi_PATEGAN"]().train(X, y)

    # Save full synthetic dataset to CSV
    output_path = "experiments/pategan/iris_synth_ui.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.generate(150).to_csv(output_path, index=False)

    return jsonify({"message": f"Synthetic data saved to {output_path}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
