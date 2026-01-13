

from __future__ import annotations

import pandas as pd
from sdv.metadata import SingleTableMetadata


def build_single_table_metadata(
    train_df: pd.DataFrame,
    target_col: str = "target"
) -> SingleTableMetadata:
    """
    Build SDV SingleTableMetadata for a single-table tabular dataset.

    - Automatically detects column types from data
    - Forces target column to categorical (classification setting)
    - Required for CopulaGAN to apply Gaussian copula transforms correctly
    """

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)

    # Ensure target is treated as categorical
    if target_col in train_df.columns:
        metadata.update_column(
            column_name=target_col,
            sdtype="categorical"
        )

    return metadata
