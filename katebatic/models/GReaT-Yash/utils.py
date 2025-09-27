import pandas as pd
import numpy as np

def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize strings, harmonize NaNs, and perform light cleanup
    on model outputs. Extend with dataset-specific mappers if needed.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan})
            )
    return out

def log_cuda_info() -> None:
    """
    Print basic CUDA/GPU info when torch is available.
    """
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())
        print(
            "Device name:",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        )
    except Exception:
        pass
