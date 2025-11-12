import os
import argparse
import pandas as pd
import numpy as np
import csv
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from katabatic.evaluate.base_evaluation import Evaluation

# Need to change this prolly


def load_data(synthetic_dir, real_test_dir):
    x_synth = pd.read_csv(os.path.join(synthetic_dir, "x_synth.csv"))
    y_synth = pd.read_csv(os.path.join(
        synthetic_dir, "y_synth.csv")).values.ravel()
    x_test = pd.read_csv(os.path.join(real_test_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(
        real_test_dir, "y_test.csv")).values.ravel()
    return x_synth, y_synth, x_test, y_test


class TSTREvaluation(Evaluation):
    def __init__(self, synthetic_dir, real_test_dir, **kwargs):
        self.synthetic_dir = synthetic_dir
        self.real_test_dir = real_test_dir

        self.x_train, self.y_train, self.x_test, self.y_test = load_data(
            synthetic_dir, real_test_dir)

    def evaluate(self):
        results = {}
        # Calculate class imbalance ratio for XGBoost
        num_neg = np.sum(self.y_train == 0)
        num_pos = np.sum(self.y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        models = {
            # "LR": LogisticRegression(max_iter=1000, random_state=42),
            # "MLP": MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True, random_state=42),
            # "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            # "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
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

            # Add AUC for binary classification
            if len(np.unique(self.y_test)) == 2:
                metrics['AUC'] = roc_auc_score(self.y_test, y_prob)

            results[name] = metrics

        self.save_results_to_csv(results, self.synthetic_dir)

        print("\nTSTR Evaluation Results:")
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
        output_path = os.path.join(results_dir, f"{model_name}_tstr.csv")

        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model", "Metric", "Value"])
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    writer.writerow([model_name, metric_name, round(value, 4)])

        print(f"\nResults saved to: {output_path}")
