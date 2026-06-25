"""Constant and near-constant column detection."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.issues import (
    CATEGORY_CONSTANTS,
    CONFIDENCE_HIGH,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    Issue,
)
from sanipy.utils.formatting import pct


def check_constant_columns(
    df: pd.DataFrame,
    config: SanipyConfig,
) -> list[Issue]:
    """Detect columns with zero or near-zero variance."""
    issues: list[Issue] = []

    if df.empty:
        return issues

    n_rows = len(df)

    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)

        # Fully constant (only 1 unique value or all-null)
        if n_unique <= 1:
            issues.append(Issue(
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
                    "sample_value": (
                        str(df[col].dropna().iloc[0])
                        if n_unique == 1 and not df[col].dropna().empty
                        else None
                    ),
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
            top_value_count = int(df[col].value_counts(dropna=True).iloc[0])
            top_frac = top_value_count / n_rows

            if top_frac >= config.near_constant_threshold:
                top_value = df[col].value_counts(dropna=True).index[0]
                issues.append(Issue(
                    id=f"near-constant-{col}",
                    title=(
                        f'Column "{col}" is near-constant --'
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
