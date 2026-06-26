"""Target column sanity checks for classification and regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_TARGET,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.text_formatting import pct


def _detect_task(
    target_series: pd.Series,
    config: SanipyConfig,
) -> str:
    """Auto-detect whether the target suggests classification or regression."""
    try:
        is_inf = np.isinf(target_series)
        series_clean = target_series[~is_inf]
    except (TypeError, ValueError):
        series_clean = target_series

    n_unique = series_clean.nunique(dropna=True)
    if n_unique <= config.auto_detect_task_max_classes:
        return "classification"
    return "regression"


def check_target_analysis(
    df: pd.DataFrame,
    target: str | None,
    task: str | None,
    config: SanipyConfig,
) -> tuple[list[DiagnosticIssue], str | None]:
    """Run target-specific sanity checks.

    Returns:
        (issues, resolved_task) where resolved_task is the final task
        type (may have been auto-detected).
    """
    issues: list[DiagnosticIssue] = []

    if target is None or target not in df.columns:
        return issues, task

    col = safe_get_series(df, target)
    n_rows = len(df)

    # Missing target values
    n_missing = int(col.isnull().sum())
    if n_missing > 0:
        frac = n_missing / n_rows
        severity = SEVERITY_CRITICAL if frac > 0.05 else SEVERITY_HIGH
        issues.append(DiagnosticIssue(
            id="target-missing",
            title=(
                f'Target column "{target}" has {n_missing:,} missing values '
                f"({pct(frac)})."
            ),
            severity=severity,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={
                "missing_count": n_missing,
                "missing_fraction": round(frac, 4),
            },
            recommendation=(
                "Rows with missing target values cannot be used for "
                "supervised learning. Drop them or investigate why "
                "they are missing."
            ),
            confidence=CONFIDENCE_HIGH,
        ))

    # Auto-detect task if not specified
    resolved_task = task
    if resolved_task is None:
        resolved_task = _detect_task(col.dropna(), config)
        issues.append(DiagnosticIssue(
            id="target-task-auto",
            title=(
                f'Task auto-detected as "{resolved_task}" based on '
                f"target column characteristics."
            ),
            severity=SEVERITY_INFO,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={
                "unique_values": int(col.nunique(dropna=True)),
                "auto_detect_max_classes": config.auto_detect_task_max_classes,
            },
            recommendation=(
                f'If this is wrong, pass task="classification" or '
                f'task="regression" explicitly.'
            ),
            confidence=CONFIDENCE_MEDIUM,
        ))

    non_null = col.dropna()
    if len(non_null) == 0:
        issues.append(DiagnosticIssue(
            id="target-all-null",
            title=f'Target column "{target}" is entirely empty/null.',
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={"n_non_null": 0},
            recommendation="Please provide a target column with non-null values for training.",
            confidence=CONFIDENCE_HIGH,
        ))
        return issues, resolved_task

    if resolved_task == "classification":
        issues.extend(_check_classification_target(non_null, target, config))
    elif resolved_task == "regression":
        issues.extend(_check_regression_target(non_null, target, config))

    return issues, resolved_task


def _check_classification_target(
    series: pd.Series,
    target: str,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Check class distribution for a classification target."""
    issues: list[DiagnosticIssue] = []

    value_counts = series.value_counts(normalize=True)
    n_classes = len(value_counts)
    if n_classes == 0:
        return issues

    majority_frac = float(value_counts.iloc[0])
    majority_class = value_counts.index[0]

    # Class distribution info
    issues.append(DiagnosticIssue(
        id="target-class-distribution",
        title=(
            f'Target "{target}" has {n_classes} classes. '
            f'Majority class "{majority_class}" represents {pct(majority_frac)}.'
        ),
        severity=SEVERITY_INFO,
        category=CATEGORY_TARGET,
        columns=[target],
        evidence={
            "n_classes": n_classes,
            "majority_class": str(majority_class),
            "majority_fraction": round(majority_frac, 4),
            "class_distribution": {
                str(k): round(v, 4) for k, v in value_counts.items()
            },
        },
        recommendation="",
        confidence=CONFIDENCE_HIGH,
    ))

    # Imbalance check
    if majority_frac >= config.imbalance_critical_threshold:
        issues.append(DiagnosticIssue(
            id="target-severe-imbalance",
            title=(
                f'Target "{target}" is severely imbalanced: '
                f'majority class is {pct(majority_frac)}.'
            ),
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={
                "majority_fraction": round(majority_frac, 4),
                "threshold": config.imbalance_critical_threshold,
            },
            recommendation=(
                "Accuracy will be misleading. Use metrics like F1-score, "
                "precision, recall, PR-AUC, or ROC-AUC. Consider "
                "stratified splitting, class weighting, oversampling "
                "(SMOTE), or undersampling."
            ),
            confidence=CONFIDENCE_HIGH,
        ))
    elif majority_frac >= config.imbalance_majority_threshold:
        issues.append(DiagnosticIssue(
            id="target-imbalance",
            title=(
                f'Target "{target}" is imbalanced: '
                f'majority class is {pct(majority_frac)}.'
            ),
            severity=SEVERITY_HIGH,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={
                "majority_fraction": round(majority_frac, 4),
                "threshold": config.imbalance_majority_threshold,
            },
            recommendation=(
                "Use stratified train/test split. Consider F1-score "
                "or ROC-AUC instead of accuracy. Class weighting "
                "may also help."
            ),
            confidence=CONFIDENCE_HIGH,
        ))

    # Binary vs multi-class info
    if n_classes == 1:
        issues.append(DiagnosticIssue(
            id="target-single-class",
            title=f'Target "{target}" has only 1 class: cannot train a classifier.',
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={"n_classes": 1},
            recommendation="A classification target must have at least 2 classes.",
            confidence=CONFIDENCE_HIGH,
        ))

    return issues


def _check_regression_target(
    series: pd.Series,
    target: str,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Check distribution of a regression target."""
    issues: list[DiagnosticIssue] = []

    # Must be numeric
    if not pd.api.types.is_numeric_dtype(series):
        issues.append(DiagnosticIssue(
            id="target-not-numeric",
            title=(
                f'Target "{target}" is not numeric (dtype: {series.dtype}). '
                "This is unusual for regression."
            ),
            severity=SEVERITY_HIGH,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={"dtype": str(series.dtype)},
            recommendation=(
                "Convert the target to a numeric type, or switch to "
                'task="classification".'
            ),
            confidence=CONFIDENCE_HIGH,
        ))
        return issues

    # Check for constant regression target
    n_unique = series.nunique()
    if n_unique <= 1:
        issues.append(DiagnosticIssue(
            id="target-constant-regression",
            title=f'Regression target "{target}" is constant (only {n_unique} unique value).',
            severity=SEVERITY_CRITICAL,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={"n_unique": n_unique},
            recommendation="A regression target should have more than 1 unique value.",
            confidence=CONFIDENCE_HIGH,
        ))
        return issues

    # Skewness calculation (excluding infinities)
    try:
        is_inf = np.isinf(series)
        series_clean = series[~is_inf]
    except (TypeError, ValueError):
        series_clean = series

    skew = series_clean.skew()
    if pd.notna(skew) and abs(float(skew)) >= config.target_skewness_threshold:
        skew = float(skew)
        # Suggest log transform only for positive values
        all_positive = bool((series_clean > 0).all()) if not series_clean.empty else False
        if all_positive:
            rec = (
                f"Target is highly skewed (skewness={skew:.2f}). "
                "Since all values are positive, consider a log-transform "
                "(np.log1p) to reduce skew."
            )
        else:
            rec = (
                f"Target is highly skewed (skewness={skew:.2f}). "
                "Consider Box-Cox, Yeo-Johnson, or robust scaling. "
                "Log-transform is not applicable because values are "
                "not all positive."
            )
        issues.append(DiagnosticIssue(
            id="target-skewness",
            title=f'Regression target "{target}" is highly skewed (skewness={skew:.2f}).',
            severity=SEVERITY_MEDIUM,
            category=CATEGORY_TARGET,
            columns=[target],
            evidence={
                "skewness": round(skew, 4),
                "threshold": config.target_skewness_threshold,
                "all_positive": all_positive,
            },
            recommendation=rec,
            confidence=CONFIDENCE_HIGH,
        ))

    # Outliers in target (IQR method using clean series fences)
    q1 = float(series_clean.quantile(0.25))
    q3 = float(series_clean.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr
        # Outliers include infinities, if present
        outlier_mask = (series < lower) | (series > upper)
        n_outliers = int(outlier_mask.sum())
        if n_outliers > 0:
            frac = n_outliers / len(series)
            issues.append(DiagnosticIssue(
                id="target-outliers",
                title=(
                    f'Regression target "{target}" has {n_outliers:,} '
                    f"possible outliers ({pct(frac)})."
                ),
                severity=SEVERITY_MEDIUM if frac > 0.01 else SEVERITY_INFO,
                category=CATEGORY_TARGET,
                columns=[target],
                evidence={
                    "outlier_count": n_outliers,
                    "outlier_fraction": round(frac, 4),
                    "lower_fence": round(lower, 4),
                    "upper_fence": round(upper, 4),
                    "iqr": round(iqr, 4),
                },
                recommendation=(
                    "Investigate extreme target values. Outlier targets "
                    "can disproportionately affect regression loss. "
                    "Consider robust loss functions or capping."
                ),
                confidence=CONFIDENCE_MEDIUM,
            ))

    return issues

