from pathlib import Path
import pandas as pd

from .core import TabPFGen


class TabPFGenModel:
    def __init__(
        self,
        task: str = "classification",
        n_sgld_steps: int = 300,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        balance_classes: bool = True,
        device: str = "auto",
        y_col: str | None = None,   
    ):
        self.task = task.lower().strip()
        self.balance_classes = bool(balance_classes)
        self.y_col = y_col  # if None -> auto based on task

        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        self.generator = TabPFGen(
            n_sgld_steps=n_sgld_steps,
            sgld_step_size=sgld_step_size,
            sgld_noise_scale=sgld_noise_scale,
            device=device,
        )

    def train(self, output_dir, *args, **kwargs):
        output_dir = Path(output_dir)

        #  robust synthetic_dir handling
        sd = kwargs.get("synthetic_dir", None)
        synthetic_dir = Path(sd) if sd else (output_dir / "synthetic")
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        #  keep column names
        X_train_df = pd.read_csv(output_dir / "x_train.csv")
        y_train = pd.read_csv(output_dir / "y_train.csv").squeeze().to_numpy()

        X_train = X_train_df.to_numpy()

        # Generate
        if self.task == "classification":
            X_syn, y_syn = self.generator.generate_classification(
                X_train,
                y_train,
                n_samples=len(y_train),
                balance_classes=self.balance_classes,
            )
            label_col = self.y_col or "class"
        else:
            X_syn, y_syn = self.generator.generate_regression(
                X_train,
                y_train,
                n_samples=len(y_train),
            )
            label_col = self.y_col or "target"

        # Save with original feature names
        x_df = pd.DataFrame(X_syn, columns=X_train_df.columns)
        y_df = pd.DataFrame({label_col: y_syn})

        x_df.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_df.to_csv(synthetic_dir / "y_synth.csv", index=False)

        full = x_df.copy()
        full[label_col] = y_syn
        full.to_csv(synthetic_dir / "synthetic.csv", index=False)

        return {
            "synthetic_csv": str(synthetic_dir / "synthetic.csv"),
            "x_synth_csv": str(synthetic_dir / "x_synth.csv"),
            "y_synth_csv": str(synthetic_dir / "y_synth.csv"),
        }
