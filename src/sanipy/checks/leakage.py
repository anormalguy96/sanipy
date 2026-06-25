"""Possible target leakage heuristics.

IMPORTANT: These are heuristic checks. They may produce false positives.
All warnings use cautious language ("possible", "may", "review").
"""

from __future__ import annotations

import re

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.issues import (
    CATEGORY_LEAKAGE,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    Issue,
)


# ── Suspicious column-name patterns ─────────────────────────────────
# These patterns, combined with high target correlation, suggest that
# the column may encode information that would not be available at
# prediction time.
_LEAKAGE_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btarget\b", re.IGNORECASE),
    re.compile(r"\blabel\b", re.IGNORECASE),
    re.compile(r"\bresult\b", re.IGNORECASE),
    re.compile(r"\boutcome\b", re.IGNORECASE),
    re.compile(r"\bfinal\b", re.IGNORECASE),
    re.compile(r"\bstatus_after\b", re.IGNORECASE),
    re.compile(r"\bpost_\b", re.IGNORECASE),
    re.compile(r"\bfuture_\b", re.IGNORECASE),
    re.compile(r"_status$", re.IGNORECASE),
    re.compile(r"_result$", re.IGNORECASE),
    re.compile(r"_outcome$", re.IGNORECASE),
    re.compile(r"closed_at$", re.IGNORECASE),
    re.compile(r"resolved_at$", re.IGNORECASE),
    re.compile(r"completed_at$", re.IGNORECASE),
    re.compile(r"ended_at$", re.IGNORECASE),
]


def _name_is_suspicious(col_name: str) -> bool:
    """Check if a column name matches leakage-risk patterns."""
    return any(p.search(col_name) for p in _LEAKAGE_NAME_PATTERNS)


def check_leakage(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[Issue]:
    """Detect columns that may cause target leakage.

    Two independent heuristics:
    1. **Name-based**: Column name matches suspicious patterns.
    2. **Correlation-based**: Extremely high correlation with target.

    When both match, confidence is higher.
    """
    issues: list[Issue] = []

    if target is None or target not in df.columns or df.empty:
        return issues

    target_series = df[target]
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)

    for col in df.columns:
        if col == target:
            continue

        suspicious_name = _name_is_suspicious(col)

        # Correlation-based check (only for numeric columns with numeric target)
        high_corr = False
        corr_val = None
        if is_numeric_target and pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr_val = float(df[col].corr(target_series))
            except (ValueError, TypeError):
                corr_val = None

            if corr_val is not None and not pd.isna(corr_val):
                if abs(corr_val) >= config.leakage_correlation_threshold:
                    high_corr = True

        # Decision matrix
        if suspicious_name and high_corr:
            issues.append(Issue(
                id=f"leakage-{col}",
                title=(
                    f'Column "{col}" may cause target leakage --'
                    f"suspicious name and very high correlation "
                    f"(r={corr_val:.3f}) with target."
                ),
                severity=SEVERITY_HIGH,
                category=CATEGORY_LEAKAGE,
                columns=[col],
                evidence={
                    "suspicious_name": True,
                    "correlation_with_target": (
                        round(corr_val, 4) if corr_val is not None else None
                    ),
                    "threshold": config.leakage_correlation_threshold,
                },
                recommendation=(
                    f'Column "{col}" has a suspicious name and extremely '
                    f'high correlation with target "{target}". '
                    "This strongly suggests it encodes post-event "
                    "information. Manually review and consider removing."
                ),
                confidence=CONFIDENCE_MEDIUM,
            ))
        elif high_corr:
            issues.append(Issue(
                id=f"leakage-corr-{col}",
                title=(
                    f'Column "{col}" has possible leakage risk --'
                    f"very high correlation (r={corr_val:.3f}) with target."
                ),
                severity=SEVERITY_MEDIUM,
                category=CATEGORY_LEAKAGE,
                columns=[col],
                evidence={
                    "suspicious_name": False,
                    "correlation_with_target": (
                        round(corr_val, 4) if corr_val is not None else None
                    ),
                    "threshold": config.leakage_correlation_threshold,
                },
                recommendation=(
                    f'Column "{col}" has an unusually high correlation '
                    f'with target "{target}". Verify that this feature '
                    "would be available at prediction time."
                ),
                confidence=CONFIDENCE_LOW,
            ))
        elif suspicious_name:
            issues.append(Issue(
                id=f"leakage-name-{col}",
                title=(
                    f'Column "{col}" has a name that may indicate '
                    f"target leakage."
                ),
                severity=SEVERITY_MEDIUM,
                category=CATEGORY_LEAKAGE,
                columns=[col],
                evidence={
                    "suspicious_name": True,
                    "correlation_with_target": (
                        round(corr_val, 4) if corr_val is not None else None
                    ),
                },
                recommendation=(
                    f'Column name "{col}" resembles a post-event or '
                    "outcome-related feature. Verify that this information "
                    "would be available at prediction time."
                ),
                confidence=CONFIDENCE_LOW,
            ))

    return issues
