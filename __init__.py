import os
from Models import register_model
from synthcity.plugins import Plugins
from sklearn.datasets import load_iris


@register_model("Vidushi_PATEGAN")
class VidushiPATEGAN:
    def __init__(self):
        # Use the tuned config you had earlier
        self.plugin = Plugins().get(
            "pategan",
            n_iter=50,
            n_teachers=5,
            teacher_template="linear",
            random_state=42,
        )

    def train(self, X, y=None):
        if y is not None:
            X["target"] = y
        self.plugin.fit(X)
        return self

    def generate(self, n=100):
        return self.plugin.generate(n).dataframe()

    def save_results(self, output_path="experiments/pategan/iris_synth.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.generate(150).to_csv(output_path, index=False)
        print(f"Saved synthetic data -> {output_path}")


# ------------------------------
# Quick smoke test (only runs when you execute this file directly)
# ------------------------------
if __name__ == "__main__":
    X, y = load_iris(as_frame=True, return_X_y=True)
    model = VidushiPATEGAN().train(X, y)
    synth = model.generate(5)
    print(synth.head())
    model.save_results()
print(" Smoke test successful!")