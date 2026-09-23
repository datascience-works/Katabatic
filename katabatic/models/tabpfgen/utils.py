"""
Utility functions for TabPFGen.

Provides simple helpers for:
- Inferring target column
- Splitting dataset into features (X) and target (y)
- One-time TabPFN authentication (Windows-safe)
"""

from typing import Tuple
import pandas as pd


def infer_target_col(
    df: pd.DataFrame,
    preferred=("class", "target", "label", "y")
) -> str:
    """
    Infer the target column from a DataFrame.

    Priority:
        class > target > label > y > last column

    Args:
        df: Input DataFrame
        preferred: Preferred column names for target

    Returns:
        Name of the target column
    """
    for col in preferred:
        if col in df.columns:
            return col

    # Fallback: use last column
    return df.columns[-1]


def split_xy(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).

    Args:
        df: Input DataFrame
        target_col: Name of the target column

    Returns:
        X: Feature DataFrame
        y: Target Series

    Raises:
        ValueError: If target column is not found
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def authenticate_tabpfn(token: str = None) -> None:
    """
    Cache a TabPFN API key so model weights can be downloaded without
    the interactive browser flow (which crashes on Windows).

    Run this once before using TabPFGen for the first time:

        from katabatic.models.tabpfgen.utils import authenticate_tabpfn
        authenticate_tabpfn()

    Or pass the token directly:

        authenticate_tabpfn("your_api_key_here")

    Get your API key at: https://ux.priorlabs.ai/account
    (register, accept the license, then copy the key)

    Args:
        token: TabPFN API key. If omitted, you will be prompted to enter it.

    Raises:
        ValueError: If no token is provided or entered.
        ImportError: If tabpfn is not installed.
        RuntimeError: If caching the token fails.
    """
    try:
        from tabpfn.browser_auth import save_token
    except ImportError:
        raise ImportError("tabpfn is not installed. Run: pip install tabpfn")

    if token is None:
        token = input(
            "\nPaste your TabPFN API key from https://ux.priorlabs.ai/account\n"
            "API key: "
        ).strip()

    if not token:
        raise ValueError("No API key provided.")

    try:
        save_token(token)
        print("TabPFN token cached. TabPFGen is ready to use.")
    except Exception as e:
        raise RuntimeError(f"Failed to cache TabPFN token: {e}") from e