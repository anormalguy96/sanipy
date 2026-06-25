"""High-cardinality categorical column detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_CARDINALITY,
    CONFIDENCE_HIGH,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.type_detection import get_categorical_columns


def check_categorical_cardinality(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[DiagnosticIssue]:
    """Detect categorical columns with many unique values."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    cat_cols = get_categorical_columns(df)

    # Use unique column labels to avoid checking duplicate columns multiple times
    unique_cat_cols = []
    seen = set()
    for col in cat_cols:
        if col not in seen:
            seen.add(col)
            unique_cat_cols.append(col)

    for col in unique_cat_cols:
        # Don't flag the target column
        if col == target:
            continue

        series = safe_get_series(df, col)
        n_unique = series.nunique(dropna=True)

        if n_unique > config.high_cardinality_threshold:
            issues.append(DiagnosticIssue(
                id=f"high-cardinality-{col}",
                title=(
                    f'Column "{col}" has high cardinality '
                    f"({n_unique} unique values)."
                ),
                severity=SEVERITY_MEDIUM,
                category=CATEGORY_CARDINALITY,
                columns=[col],
                evidence={
                    "unique_values": n_unique,
                    "threshold": config.high_cardinality_threshold,
                    "dtype": str(series.dtype),
                },
                recommendation=(
                    f'Column "{col}" has {n_unique} unique values. '
                    "One-hot encoding would create too many features. "
                    "Consider target encoding, frequency encoding, "
                    "hashing, or grouping rare categories."
                ),
                confidence=CONFIDENCE_HIGH,
            ))

    return issues

