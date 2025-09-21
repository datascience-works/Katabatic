# katebatic/models/great/sick_models.py

import pandas as pd
from scipy.io import arff
import torch
from be_great import GReaT
from transformers import EarlyStoppingCallback


def load_sick_dataset(arff_path: str) -> pd.DataFrame:
    """Load the Sick dataset from ARFF and decode bytes to strings."""
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df


def train_great_on_sick(df: pd.DataFrame, epochs: int = 3, model_name: str = "distilgpt2") -> GReaT:
    """Train GReaT on the Sick dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GReaT(
        llm=model_name,
        batch_size=16,
        epochs=epochs,
        fp16=True if device.type == "cuda" else False,
        dataloader_num_workers=2,
    )

    # Configure evaluation/early stopping
    if hasattr(model, "validation_split"):
        model.validation_split = 0.15
    elif hasattr(model, "eval_dataset_size"):
        model.eval_dataset_size = 0.15

    if hasattr(model, "evaluation_strategy"):
        model.evaluation_strategy = "epoch"
    if hasattr(model, "save_strategy"):
        model.save_strategy = "epoch"
    if hasattr(model, "logging_strategy"):
        model.logging_strategy = "epoch"

    if hasattr(model, "load_best_model_at_end"):
        model.load_best_model_at_end = True
    if hasattr(model, "metric_for_best_model"):
        model.metric_for_best_model = "eval_loss"
    if hasattr(model, "greater_is_better"):
        model.greater_is_better = False

    if hasattr(model, "early_stopping_patience"):
        model.early_stopping_patience = 2

    if hasattr(model, "callbacks"):
        if model.callbacks is None:
            model.callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        else:
            model.callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    print("🚀 Training GReaT on Sick dataset...")
    model.fit(df)
    return model
