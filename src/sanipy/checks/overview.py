"""Basic dataset overview check."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.issues import (
    CATEGORY_OVERVIEW,
    CONFIDENCE_HIGH,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    Issue,
)
from sanipy.utils.dtype import column_type_summary


def check_overview(
    df: pd.DataFrame,
    target: str | None,
    config: SanipyConfig,
) -> tuple[list[Issue], dict]:
    """Produce dataset overview info and check for basic problems.

    Returns:
        A tuple of (issues, dataset_info_dict).
    """
    issues: list[Issue] = []

    n_rows, n_cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    type_counts = column_type_summary(df)

    dataset_info = {
        "rows": n_rows,
        "columns": n_cols,
        "memory_mb": memory_mb,
        "column_types": type_counts,
    }

    # Empty dataset
    if n_rows == 0:
        issues.append(Issue(
            id="overview-001",
            title="Dataset is empty (0 rows).",
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_OVERVIEW,
            evidence={"rows": 0},
            recommendation="Provide a non-empty dataset.",
            confidence=CONFIDENCE_HIGH,
        ))
        return issues, dataset_info

    # Very small dataset
    if n_rows < 30:
        issues.append(Issue(
            id="overview-002",
            title=f"Dataset is very small ({n_rows} rows).",
            severity=SEVERITY_INFO,
            category=CATEGORY_OVERVIEW,
            evidence={"rows": n_rows},
            recommendation=(
                "Results on very small datasets may not generalize. "
                "Consider collecting more data."
            ),
            confidence=CONFIDENCE_HIGH,
        ))

    # Target column existence
    if target is not None and target not in df.columns:
        issues.append(Issue(
            id="overview-003",
            title=f'Target column "{target}" not found in the dataset.',
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_OVERVIEW,
            columns=[target],
            evidence={"available_columns": df.columns.tolist()[:20]},
            recommendation=(
                "Check the target column name for typos. "
                f"Available columns: {', '.join(df.columns.tolist()[:10])}..."
            ),
            confidence=CONFIDENCE_HIGH,
        ))

    # No columns
    if n_cols == 0:
        issues.append(Issue(
            id="overview-004",
            title="Dataset has no columns.",
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_OVERVIEW,
            evidence={"columns": 0},
            recommendation="Provide a dataset with at least one column.",
            confidence=CONFIDENCE_HIGH,
        ))

    return issues, dataset_info
