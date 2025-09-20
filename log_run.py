import argparse, datetime, os, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TEMPLATE = """# PATE-GAN — Iris (UI run)
- **Model:** {model}
- **Dataset:** sklearn Iris (150 rows, 4 features + target)
- **Params:** {params}
- **Generated:** {n_rows} rows
- **Artifacts**
  - CSV: `{csv_path}`
- **Metrics**
  - Train-on-synth, test-on-real accuracy: {acc:.3f}
- **Notes:** {notes}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/pategan/iris_synth_ui.csv")
    ap.add_argument("--model", default="Vidushi_PATEGAN")
    ap.add_argument("--params", default='n_iter=5, n_teachers=3, teacher_template="linear", random_state=42')
    ap.add_argument("--notes", default="Smoke test via `/pategan/generate` and `/pategan/save` succeeded.")
    args = ap.parse_args()

    # Load synth
    df_synth = pd.read_csv(args.csv)
    n_rows = len(df_synth)

    # Quick smoke metric (train on synth, test on real)
    X_real, y_real = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.30, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(df_synth.drop(columns=["target"]), df_synth["target"])
    acc = accuracy_score(y_test, clf.predict(X_test))

    # Write markdown
    today = datetime.date.today().isoformat()
    out_dir = "experiments/pategan"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{today}.md")

    content = TEMPLATE.format(
        model=args.model,
        params=args.params,
        n_rows=n_rows,
        csv_path=args.csv,
        acc=acc,
        notes=args.notes,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Wrote experiment log -> {out_path}")

if __name__ == "__main__":
    main()
    print(" Log run successful!")