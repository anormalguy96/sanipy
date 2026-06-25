"""Private DataFrame helper operations for Sanipy."""

from __future__ import annotations

import pandas as pd


def get_df_memory_usage_mb(df: pd.DataFrame) -> float:
    """Return memory usage of the DataFrame in MB."""
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)


def safe_get_series(df: pd.DataFrame, col: str | int | float | bool | tuple) -> pd.Series:
    """Extract a Series from the DataFrame for column 'col', handling duplicate columns safely."""
    col_data = df[col]
    if isinstance(col_data, pd.DataFrame):
        # If duplicate, take the first column
        return col_data.iloc[:, 0]
    return col_data
