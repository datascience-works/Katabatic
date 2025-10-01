import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(synthetic_df, real_df):
    """
    Preprocess synthetic and real datasets:
    - One-hot encode categorical features
    - Align columns between datasets
    - Scale features
    - Encode target labels
    """

    # Separate features and target
    X_syn = synthetic_df.drop(columns=["Class"])
    y_syn = synthetic_df["Class"]

    categorical_cols = X_syn.select_dtypes(include=['object']).columns.tolist()
    X_syn_encoded = pd.get_dummies(X_syn, columns=categorical_cols)
    X_real_encoded = pd.get_dummies(real_df.drop(columns=["Class"]), columns=categorical_cols)

    # Align columns
    X_real_encoded = X_real_encoded.reindex(columns=X_syn_encoded.columns, fill_value=0)

    # Fill NaNs
    X_syn_encoded = X_syn_encoded.fillna(0)
    X_real_encoded = X_real_encoded.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_syn_scaled = scaler.fit_transform(X_syn_encoded)
    X_real_scaled = scaler.transform(X_real_encoded)

    # Encode target labels
    le = LabelEncoder()
    y_syn_encoded = le.fit_transform(y_syn)
    y_real_encoded = le.transform(real_df["Class"])

    classes = np.unique(y_syn_encoded)

    return X_syn_scaled, y_syn_encoded, X_real_scaled, y_real_encoded, classes
