"""Numeric outlier detection using the IQR method."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_OUTLIERS,
    CONFIDENCE_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
import numpy as np
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.type_detection import get_numeric_columns
from sanipy._utils.text_formatting import pct


def check_outlier_detection(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[DiagnosticIssue]:
    """Detect numeric columns with suspicious outliers (IQR method)."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    numeric_cols = get_numeric_columns(df)
    n_rows = len(df)

    # Use unique column labels to avoid checking duplicate columns multiple times
    unique_numeric_cols = []
    seen = set()
    for col in numeric_cols:
        if col not in seen:
            seen.add(col)
            unique_numeric_cols.append(col)

    for col in unique_numeric_cols:
        # Skip target — handled by target checks
        if col == target:
            continue

        series = safe_get_series(df, col)
        series_clean = series.dropna()

        # Filter out infinities for fence calculations
        try:
            is_inf = np.isinf(series_clean)
            series_fences = series_clean[~is_inf]
        except (TypeError, ValueError):
            series_fences = series_clean

        if len(series_fences) < 10:
            continue

        try:
            q1 = float(series_fences.quantile(0.25))
            q3 = float(series_fences.quantile(0.75))
        except (TypeError, ValueError):
            continue

        iqr = q3 - q1

        if iqr == 0:
            continue  # No spread — handled by constant column check

        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr

        outlier_mask = (series_clean < lower) | (series_clean > upper)
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

        try:
            val_min = float(series_clean.min())
            val_max = float(series_clean.max())
            evidence_min = round(val_min, 4) if not np.isinf(val_min) and not np.isnan(val_min) else str(val_min)
            evidence_max = round(val_max, 4) if not np.isinf(val_max) and not np.isnan(val_max) else str(val_max)
        except (TypeError, ValueError):
            evidence_min = None
            evidence_max = None

        issues.append(DiagnosticIssue(
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
                "min": evidence_min,
                "max": evidence_max,
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

