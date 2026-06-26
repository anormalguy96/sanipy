"""Constant and near-constant column detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_CONSTANTS,
    CONFIDENCE_HIGH,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.text_formatting import pct


def check_constant_features(
    df: pd.DataFrame,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect columns with zero or near-zero variance."""
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
        n_unique = series.nunique(dropna=True)

        # Fully constant (only 1 unique value or all-null)
        if n_unique <= 1:
            sample_val = None
            if n_unique == 1:
                dropped = series.dropna()
                if not dropped.empty:
                    sample_val = str(dropped.iloc[0])

            issues.append(DiagnosticIssue(
                id=f"constant-{col}",
                title=(
                    f'Column "{col}" is constant '
                    f"({n_unique} unique value{'s' if n_unique != 1 else ''})."
                ),
                severity=SEVERITY_HIGH,
                category=CATEGORY_CONSTANTS,
                columns=[col],
                evidence={
                    "unique_values": n_unique,
                    "sample_value": sample_val,
                },
                recommendation=(
                    f'Column "{col}" provides no information for ML. '
                    "Consider removing it."
                ),
                confidence=CONFIDENCE_HIGH,
            ))
            continue

        # Near-constant: one value dominates
        if n_rows > 0:
            val_counts = series.value_counts(dropna=True)
            if not val_counts.empty:
                top_value_count = int(val_counts.iloc[0])
                top_frac = top_value_count / n_rows

                if top_frac >= config.near_constant_threshold:
                    top_value = val_counts.index[0]
                    issues.append(DiagnosticIssue(
                        id=f"near-constant-{col}",
                        title=(
                            f'Column "{col}" is near-constant: '
                            f'value "{top_value}" appears in {pct(top_frac)} of rows.'
                        ),
                        severity=SEVERITY_MEDIUM,
                        category=CATEGORY_CONSTANTS,
                        columns=[col],
                        evidence={
                            "dominant_value": str(top_value),
                            "dominant_fraction": round(top_frac, 4),
                            "unique_values": n_unique,
                        },
                        recommendation=(
                            f'Column "{col}" has extremely low variance. '
                            "It may not contribute useful information for ML. "
                            "Review whether it should be kept."
                        ),
                        confidence=CONFIDENCE_HIGH,
                    ))

    return issues

