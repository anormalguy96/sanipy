"""Private DataFrame helper operations for Sanipy."""

from __future__ import annotations

import pandas as pd


def get_df_memory_usage_mb(df: pd.DataFrame) -> float:
    """Return memory usage of the DataFrame in MB."""
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
