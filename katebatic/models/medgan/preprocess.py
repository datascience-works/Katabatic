# katabatic/models/medgan/preprocess.py
from pathlib import Path
from typing import Tuple
try:  # Python 3.8+
    from typing import Literal
except ImportError:  # Python 3.7 and below
    from typing_extensions import Literal
import numpy as np
import pandas as pd

DataType = Literal["binary", "count"]

def to_matrix_file(
    X, out_path: str, data_type: DataType = "binary"
) -> Tuple[str, int]:
    """
    Convert a DataFrame/ndarray to medGAN's expected multi-hot matrix file.
    Returns (matrix_path, n_features).
    """
    arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features).")
    if data_type == "binary":
        # ensure [0,1] with ints
        arr = (arr > 0.5).astype(np.int8)
    elif data_type == "count":
        arr = np.maximum(arr, 0).astype(np.int32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # medGAN accepts .matrix; we store as .npy for safety (works in practice)
    np.save(out_path, arr, allow_pickle=False)
    return out_path, arr.shape[1]
