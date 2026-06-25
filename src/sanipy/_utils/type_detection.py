"""Type-detection helpers for Sanipy checks."""

from __future__ import annotations

import pandas as pd


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return names of numeric (int/float) columns."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return names of object / category / boolean columns."""
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return names of datetime columns."""
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def column_type_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return a {type_label: count} summary of column types."""
    summary: dict[str, int] = {}
    for dtype_name in df.dtypes:
        kind = dtype_name.kind  # type: ignore[union-attr]
        label = {
            "i": "integer",
            "u": "unsigned_integer",
            "f": "float",
            "b": "boolean",
            "O": "object",
            "S": "bytes",
            "U": "string",
            "M": "datetime",
            "m": "timedelta",
        }.get(kind, "other")
        summary[label] = summary.get(label, 0) + 1
    return summary
