from __future__ import annotations

import pandas as pd
from sdv.metadata import SingleTableMetadata


def build_single_table_metadata(
    train_df: pd.DataFrame,
    target_col: str = "target"
) -> SingleTableMetadata:
    """
    Build metadata for CopulaGAN.

    Automatically detects column types.
    Target column type is inferred:
    - categorical if few unique values (classification)
    - numerical otherwise (regression)
    """

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)

    if target_col in train_df.columns:
        unique_values = train_df[target_col].nunique()

        # heuristic to decide type
        if unique_values <= 20:
            metadata.update_column(
                column_name=target_col,
                sdtype="categorical"
            )
        else:
            metadata.update_column(
                column_name=target_col,
                sdtype="numerical"
            )

    return metadata
