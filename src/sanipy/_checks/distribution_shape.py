"""Numeric feature skewness detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_SKEWNESS,
    CONFIDENCE_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.type_detection import get_numeric_columns


def check_distribution_shape(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[DiagnosticIssue]:
    """Detect highly skewed numeric columns."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    numeric_cols = get_numeric_columns(df)

    for col in numeric_cols:
        # Skip target — handled by target checks
        if col == target:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        skew = float(series.skew())
        abs_skew = abs(skew)

        if abs_skew < config.skewness_warn_threshold:
            continue

        severity = SEVERITY_MEDIUM if abs_skew >= 5.0 else SEVERITY_INFO

        # Suggest log-transform only if all values are positive
        all_positive = bool((series > 0).all())
        if all_positive:
            rec = (
                f'Column "{col}" is highly skewed (skewness={skew:.2f}). '
                "Since all values are positive, consider a log-transform "
                "(np.log1p) to reduce skew."
            )
        else:
            rec = (
                f'Column "{col}" is highly skewed (skewness={skew:.2f}). '
                "Consider robust scaling, Box-Cox (positive only), "
                "Yeo-Johnson, or rank-based transformations."
            )

        issues.append(DiagnosticIssue(
            id=f"skewness-{col}",
            title=f'Column "{col}" is highly skewed (skewness={skew:.2f}).',
            severity=severity,
            category=CATEGORY_SKEWNESS,
            columns=[col],
            evidence={
                "skewness": round(skew, 4),
                "threshold": config.skewness_warn_threshold,
                "all_positive": all_positive,
                "n_values": len(series),
            },
            recommendation=rec,
            confidence=CONFIDENCE_HIGH,
        ))

    return issues
