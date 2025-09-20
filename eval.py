import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Real data
X_real, y_real = load_iris(as_frame=True, return_X_y=True)
X_real["target"] = y_real

# Synth data from your UI save
synth_path = "experiments/pategan/iris_synth_ui.csv"
df_synth = pd.read_csv(synth_path)

# Split real test set
Xtr, Xte, ytr, yte = train_test_split(
    X_real.drop(columns=["target"]), X_real["target"], test_size=0.3, random_state=42
)

# Train on synthetic, test on real
clf = LogisticRegression(max_iter=1000)
clf.fit(df_synth.drop(columns=["target"]), df_synth["target"])
yhat = clf.predict(Xte)
acc = accuracy_score(yte, yhat)

print(f"[PATE-GAN] Train-on-synth, test-on-real accuracy: {acc:.3f}")
print(" Eval script successful!")