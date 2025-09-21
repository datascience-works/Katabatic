# katebatic/models/great/sick_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


def save_dataset(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ Saved dataset at {path} (shape={df.shape})")


def evaluate_models(X_train, X_test, y_train, y_test, task="classification"):
    """Train and evaluate models for classification or regression tasks."""
    assert task in ["classification", "regression"]

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
        }

    for name, model in models.items():
        print(f"\n=== {name} ===")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), X_train.select_dtypes(include=[np.number]).columns),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), X_train.select_dtypes(exclude=[np.number]).columns)
            ]
        )

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if task == "classification":
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
            print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
            print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
        else:
            print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
            print(f"R²: {r2_score(y_test, y_pred):.4f}")
