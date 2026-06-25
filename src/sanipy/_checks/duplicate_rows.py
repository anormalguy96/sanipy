"""Duplicate row detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_DUPLICATES,
    CONFIDENCE_HIGH,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.text_formatting import pct


def check_duplicate_rows(
    df: pd.DataFrame,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect fully duplicate rows."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    n_rows = len(df)
    n_duplicates = int(df.duplicated().sum())

    if n_duplicates == 0:
        return issues

    frac = n_duplicates / n_rows

    if frac >= 0.10:
        severity = SEVERITY_HIGH
    elif frac >= config.duplicate_warn_threshold:
        severity = SEVERITY_MEDIUM
    else:
        severity = SEVERITY_INFO

    issues.append(DiagnosticIssue(
        id="duplicates-001",
        title=(
            f"Dataset contains {n_duplicates:,} duplicate rows "
            f"({pct(frac)} of total)."
        ),
        severity=severity,
        category=CATEGORY_DUPLICATES,
        evidence={
            "duplicate_count": n_duplicates,
            "duplicate_fraction": round(frac, 4),
            "total_rows": n_rows,
        },
        recommendation=(
            "Review whether duplicates are expected (e.g., repeated "
            "transactions) or accidental. Consider removing duplicates "
            "before training to avoid data leakage and biased evaluation."
        ),
        confidence=CONFIDENCE_HIGH,
    ))

    return issues
