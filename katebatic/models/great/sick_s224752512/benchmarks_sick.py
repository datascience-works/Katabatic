# katebatic/models/great/benchmarks_sick.py

from .sick_models import load_sick_dataset, train_great_on_sick
from .sick_utils import save_dataset, evaluate_models
import pandas as pd
from sklearn.model_selection import train_test_split


def run_sick_pipeline():
    # 1. Load ARFF dataset
    arff_path = "dataset_38_sick.arff"
    df = load_sick_dataset(arff_path)
    print("✅ Loaded Sick dataset:", df.shape)

    # Save real dataset
    save_dataset(df, "real_sick.csv")

    # 2. Train GReaT model
    model = train_great_on_sick(df, epochs=3)

    # 3. Generate synthetic data
    print("🧪 Generating synthetic samples...")
    try:
        synthetic_data = model.sample(n_samples=50, temperature=0.7, guided_sampling=True)
    except RuntimeError:
        print("⚠️ GPU unavailable, switching to CPU for sampling...")
        model.device = "cpu"
        synthetic_data = model.sample(n_samples=20, temperature=0.7, guided_sampling=True)

    save_dataset(synthetic_data, "synthetic_sick.csv")

    # 4. Benchmark: Train/Test on Real vs Synthetic
    target = "Class"
    X_real, y_real = df.drop(columns=[target]), df[target]
    X_synth, y_synth = synthetic_data.drop(columns=[target]), synthetic_data[target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)

    print("\n--- Train on Real, Test on Real ---")
    evaluate_models(Xr_train, Xr_test, yr_train, yr_test, task="classification")

    print("\n--- Train on Synthetic, Test on Real ---")
    evaluate_models(Xs_train, Xr_test, ys_train, yr_test, task="classification")


if __name__ == "__main__":
    run_sick_pipeline()
