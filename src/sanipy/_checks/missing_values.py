"""Missing value detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_MISSING,
    CONFIDENCE_HIGH,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.text_formatting import pct


def check_missing_values(
    df: pd.DataFrame,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect columns with missing values and assign severity."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    n_rows = len(df)

    # Use unique column labels to avoid checking duplicate columns multiple times
    unique_cols = []
    seen = set()
    for col in df.columns:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)

    for col in unique_cols:
        series = safe_get_series(df, col)
        n_missing = int(series.isnull().sum())
        if n_missing == 0:
            continue

        frac = n_missing / n_rows

        # Determine severity
        if frac >= config.missing_critical_threshold:
            severity = SEVERITY_CRITICAL
        elif frac >= config.missing_high_threshold:
            severity = SEVERITY_HIGH
        elif frac >= config.missing_medium_threshold:
            severity = SEVERITY_MEDIUM
        elif frac >= config.missing_low_threshold:
            severity = SEVERITY_LOW
        else:
            severity = SEVERITY_INFO

        # Build recommendation
        if frac >= config.missing_critical_threshold:
            rec = (
                f"Column \"{col}\" is almost entirely missing. "
                "Consider dropping it unless the remaining values are critical."
            )
        elif frac >= config.missing_high_threshold:
            rec = (
                f"Consider careful imputation or dropping \"{col}\". "
                "Investigate why so many values are missing."
            )
        elif frac >= config.missing_medium_threshold:
            rec = (
                f"Impute missing values in \"{col}\" (mean/median for numeric, "
                "mode for categorical) or use a model that handles nulls."
            )
        else:
            rec = (
                f"A small fraction of \"{col}\" is missing. "
                "Simple imputation or dropping rows may be acceptable."
            )

        issues.append(DiagnosticIssue(
            id=f"missing-{col}",
            title=f'Column "{col}" has {pct(frac)} missing values ({n_missing:,}/{n_rows:,}).',
            severity=severity,
            category=CATEGORY_MISSING,
            columns=[col],
            evidence={
                "missing_count": n_missing,
                "missing_fraction": round(frac, 4),
                "total_rows": n_rows,
            },
            recommendation=rec,
            confidence=CONFIDENCE_HIGH,
        ))

    return issues

