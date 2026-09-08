from pathlib import Path

import pandas as pd


def load_training_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Load training data from a Katabatic dataset directory.

    The function first looks for train_full.csv. If that file does not
    exist, it falls back to x_train.csv and y_train.csv and combines them
    into a single DataFrame.

    Parameters
    ----------
    data_dir:
        Directory containing the training data files.

    Returns
    -------
    pd.DataFrame
        Combined training dataset.

    Raises
    ------
    FileNotFoundError
        If neither supported training-data layout is found.
    ValueError
        If y_train.csv contains more than one target column.
    """
    data_dir = Path(data_dir)

    train_full_path = data_dir / "train_full.csv"

    if train_full_path.exists():
        return pd.read_csv(train_full_path)

    x_train_path = data_dir / "x_train.csv"
    y_train_path = data_dir / "y_train.csv"

    if x_train_path.exists() and y_train_path.exists():
        x_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path)

        if y_train.shape[1] != 1:
            raise ValueError("y_train.csv must contain exactly one target column.")

        return pd.concat(
            [x_train.reset_index(drop=True), y_train.reset_index(drop=True)],
            axis=1,
        )

    raise FileNotFoundError(
        "Could not find training data. Expected either train_full.csv "
        "or both x_train.csv and y_train.csv."
    )


def resolve_synth_dir(
    synthetic_dir: str | Path | None,
    data_dir: str | Path,
    model_name: str = "realtabformer",
) -> Path:
    """
    Resolve the directory used to store generated synthetic data.
    """
    if synthetic_dir is not None:
        synth_dir = Path(synthetic_dir)
    else:
        data_path = Path(data_dir)
        dataset_name = data_path.name
        synth_dir = Path("synthetic") / dataset_name / model_name

    synth_dir.mkdir(parents=True, exist_ok=True)

    return synth_dir


def save_synthetic_data(
    synthetic_df: pd.DataFrame,
    synthetic_dir: str | Path,
    target_column: str,
) -> None:
    """
    Save synthetic features and target values using Katabatic's
    standard output format.
    """
    synthetic_dir = Path(synthetic_dir)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    if target_column not in synthetic_df.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from synthetic data."
        )

    x_synth = synthetic_df.drop(columns=[target_column])
    y_synth = synthetic_df[[target_column]]

    x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
    y_synth.to_csv(synthetic_dir / "y_synth.csv", index=False)


def save_metadata(
    synthetic_dir: str | Path,
    *,
    target_column: str,
    row_count: int,
    model_parameters: dict,
) -> None:
    """
    Save basic metadata for a REaLTabFormer training run.
    """
    import json

    synthetic_dir = Path(synthetic_dir)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "realtabformer",
        "target_column": target_column,
        "synthetic_rows": row_count,
        "parameters": model_parameters,
    }

    with (synthetic_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
