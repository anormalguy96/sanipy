"""Numeric outlier detection using the IQR method."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.issues import (
    CATEGORY_OUTLIERS,
    CONFIDENCE_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    Issue,
)
from sanipy.utils.dtype import get_numeric_columns
from sanipy.utils.formatting import pct


def check_outliers(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[Issue]:
    """Detect numeric columns with suspicious outliers (IQR method)."""
    issues: list[Issue] = []

    if df.empty:
        return issues

    numeric_cols = get_numeric_columns(df)
    n_rows = len(df)

    for col in numeric_cols:
        # Skip target — handled by target checks
        if col == target:
            continue

        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue  # No spread — handled by constant column check

        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = int(outlier_mask.sum())

        if n_outliers == 0:
            continue

        frac = n_outliers / n_rows

        if frac >= 0.10:
            severity = SEVERITY_HIGH
        elif frac >= config.outlier_warn_threshold:
            severity = SEVERITY_MEDIUM
        else:
            severity = SEVERITY_INFO

        issues.append(Issue(
            id=f"outlier-{col}",
            title=(
                f'Column "{col}" contains {n_outliers:,} possible outliers '
                f"({pct(frac)}) outside IQR fences."
            ),
            severity=severity,
            category=CATEGORY_OUTLIERS,
            columns=[col],
            evidence={
                "outlier_count": n_outliers,
                "outlier_fraction": round(frac, 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            },
            recommendation=(
                f'Investigate outliers in "{col}". They may be data errors, '
                "rare but valid values, or measurement artifacts. "
                "Consider capping, Winsorizing, or using robust models. "
                "Do not automatically remove without understanding."
            ),
            confidence=CONFIDENCE_MEDIUM,
        ))

    return issues
