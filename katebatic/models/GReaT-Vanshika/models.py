import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(X_syn_scaled, y_syn_encoded, X_real_scaled, y_real_encoded, classes, seeds=[0, 1, 2, 3, 4]):
    """
    Train models (Logistic Regression, Decision Tree, Random Forest) on synthetic data
    and evaluate on real data using multiple seeds.
    """

    results = {"LR": {"accuracy": [], "f1": [], "rocauc": []},
               "DT": {"accuracy": [], "f1": [], "rocauc": []},
               "RF": {"accuracy": [], "f1": [], "rocauc": []}}

    for seed in seeds:
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real_scaled, y_real_encoded, test_size=0.2, random_state=seed, stratify=y_real_encoded
        )

        for name, model in [
            ("LR", LogisticRegression(max_iter=2000, solver='lbfgs')),
            ("DT", DecisionTreeClassifier()),
            ("RF", RandomForestClassifier())
        ]:
            # Train on synthetic data
            model.fit(X_syn_scaled, y_syn_encoded)

            # Predictions
            y_pred = model.predict(X_test_real)
            results[name]["accuracy"].append(accuracy_score(y_test_real, y_pred))
            results[name]["f1"].append(f1_score(y_test_real, y_pred, average='weighted'))

            # ROC-AUC (multiclass)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_real)
                results[name]["rocauc"].append(
                    roc_auc_score(y_test_real, y_prob, multi_class='ovr', labels=classes)
                )
            else:
                results[name]["rocauc"].append(np.nan)

    # Print average metrics
    for model in results:
        print(f"\n{model} Average Metrics:")
        for metric in results[model]:
            print(f"{metric}: {np.nanmean(results[model][metric]):.4f}")

    return results
