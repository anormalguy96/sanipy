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
import numpy as np
from sanipy._utils.dataframe_ops import safe_get_series
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

        # Safely filter out infinite values
        try:
            is_inf = np.isinf(series_clean)
            series_clean = series_clean[~is_inf]
        except (TypeError, ValueError):
            pass

        if len(series_clean) < 10:
            continue

        try:
            skew = series_clean.skew()
        except (TypeError, ValueError):
            continue

        if pd.isna(skew):
            continue

        skew = float(skew)
        abs_skew = abs(skew)

        if abs_skew < config.skewness_warn_threshold:
            continue

        severity = SEVERITY_MEDIUM if abs_skew >= 5.0 else SEVERITY_INFO

        # Suggest log-transform only if all values are positive
        all_positive = bool((series_clean > 0).all()) if not series_clean.empty else False
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
                "n_values": len(series_clean),
            },
            recommendation=rec,
            confidence=CONFIDENCE_HIGH,
        ))

    return issues

