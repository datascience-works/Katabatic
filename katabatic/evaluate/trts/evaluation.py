import os
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from katabatic.evaluate.base_evaluation import Evaluation


def load_real_data(output_dir, synthetic_dir):
    x_train = pd.read_csv(os.path.join(output_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(
        output_dir, "y_train.csv")).values.ravel()
    x_synth = pd.read_csv(os.path.join(synthetic_dir, "x_synth.csv"))
    y_synth = pd.read_csv(os.path.join(
        synthetic_dir, "y_synth.csv")).values.ravel()
    return x_train, y_train, x_synth, y_synth


class TRTSEvaluation(Evaluation):
    """
    Train on Real, Test on Synthetic.

    Trains each classifier on the real training data and evaluates it on the
    synthetic data. Measures how well the synthetic distribution matches the
    real distribution — if a model trained on real data generalises to
    synthetic data, the two distributions are statistically similar.
    """

    def __init__(self, output_dir, synthetic_dir, **kwargs):
        self.output_dir = output_dir
        self.synthetic_dir = synthetic_dir

        self.x_train, self.y_train, self.x_test, self.y_test = load_real_data(
            output_dir, synthetic_dir)

    def evaluate(self):
        results = {}
        num_neg = np.sum(self.y_train == 0)
        num_pos = np.sum(self.y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        models = {
            "LR": LogisticRegression(),
            "MLP": MLPClassifier(),
            "RF": RandomForestClassifier(),
            "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight)
        }

        for name, model in models.items():
            if name in ["LR", "MLP"]:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(self.x_train)
                x_test_scaled = scaler.transform(self.x_test)
                model.fit(x_train_scaled, self.y_train)
                y_pred = model.predict(x_test_scaled)
                y_prob = model.predict_proba(x_test_scaled)[:, 1]
            else:
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                y_prob = model.predict_proba(self.x_test)[:, 1]

            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred, average='weighted')
            }

            if len(np.unique(self.y_test)) == 2:
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)

            results[name] = metrics

        self.save_results_to_csv(results, self.synthetic_dir)

        print("\nTRTS Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        return results

    @staticmethod
    def save_results_to_csv(results, synthetic_dir):
        parts = os.path.normpath(synthetic_dir).split(os.sep)
        model_name = parts[-1]
        dataset_name = parts[-2]

        results_dir = os.path.join("Results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"{model_name}_trts.csv")

        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Metric", "Value"])
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    writer.writerow([model_name, metric_name, round(value, 4)])

        print(f"\nResults saved to: {output_path}")
