"""Correlation checks — feature-feature and feature-target."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_CORRELATION,
    CATEGORY_PERFORMANCE,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.type_detection import get_numeric_columns


def check_correlations(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
    task: str | None = None,
) -> list[DiagnosticIssue]:
    """Detect highly correlated feature pairs and feature-target relationships."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    numeric_cols = get_numeric_columns(df)

    # Remove target from feature list (we handle it separately)
    feature_cols = [c for c in numeric_cols if c != target]

    if len(feature_cols) < 2:
        return issues

    # ── Guard: too many columns ──────────────────────────────────
    if len(feature_cols) > config.max_columns_for_correlation:
        issues.append(DiagnosticIssue(
            id="correlation-skipped",
            title=(
                f"Correlation check skipped: dataset has "
                f"{len(feature_cols)} numeric columns (limit: "
                f"{config.max_columns_for_correlation})."
            ),
            severity=SEVERITY_INFO,
            category=CATEGORY_PERFORMANCE,
            evidence={
                "numeric_columns": len(feature_cols),
                "limit": config.max_columns_for_correlation,
            },
            recommendation=(
                "Increase max_columns_for_correlation in SanipyConfig "
                "if you want to run this check on wide datasets."
            ),
            confidence=CONFIDENCE_HIGH,
        ))
        return issues

    # ── Feature-feature correlation ──────────────────────────────
    corr_matrix = df[feature_cols].corr(method="pearson")

    seen_pairs: set[tuple[str, str]] = set()
    for i, col_a in enumerate(feature_cols):
        for col_b in feature_cols[i + 1:]:
            pair = (col_a, col_b)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            corr_val = corr_matrix.loc[col_a, col_b]
            if pd.isna(corr_val):
                continue

            abs_corr = abs(float(corr_val))
            if abs_corr >= config.high_correlation_threshold:
                issues.append(DiagnosticIssue(
                    id=f"correlation-{col_a}-{col_b}",
                    title=(
                        f'Features "{col_a}" and "{col_b}" are highly '
                        f"correlated (r={corr_val:.3f})."
                    ),
                    severity=SEVERITY_MEDIUM,
                    category=CATEGORY_CORRELATION,
                    columns=[col_a, col_b],
                    evidence={
                        "correlation": round(float(corr_val), 4),
                        "threshold": config.high_correlation_threshold,
                    },
                    recommendation=(
                        f'Consider removing one of "{col_a}" or '
                        f'"{col_b}" to reduce multicollinearity, '
                        "or use regularization."
                    ),
                    confidence=CONFIDENCE_HIGH,
                ))

    # ── Feature-target correlation (regression only) ─────────────
    if (
        target is not None
        and target in df.columns
        and task == "regression"
        and pd.api.types.is_numeric_dtype(df[target])
    ):
        target_series = df[target]
        for col in feature_cols:
            try:
                corr_val = float(df[col].corr(target_series))
            except (ValueError, TypeError):
                continue
            if pd.isna(corr_val):
                continue

            abs_corr = abs(corr_val)

            # Very high — possible leakage (handled by leakage module,
            # but we note it here too)
            if abs_corr >= config.leakage_correlation_threshold:
                continue  # Leakage module handles this

            # Moderate-to-high correlation: informational
            if abs_corr >= 0.7:
                issues.append(DiagnosticIssue(
                    id=f"target-corr-{col}",
                    title=(
                        f'Feature "{col}" has strong correlation with '
                        f'target "{target}" (r={corr_val:.3f}).'
                    ),
                    severity=SEVERITY_INFO,
                    category=CATEGORY_CORRELATION,
                    columns=[col, target],
                    evidence={
                        "correlation": round(corr_val, 4),
                    },
                    recommendation=(
                        f'"{col}" is strongly correlated with the target. '
                        "This may be a useful feature -- verify it is not "
                        "leaking future information."
                    ),
                    confidence=CONFIDENCE_MEDIUM,
                ))

    return issues
