"""Private orchestration engine for running Sanipy dataset checks."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_PERFORMANCE,
    CONFIDENCE_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    DiagnosticIssue,
)
from sanipy.reports import DatasetReport
from sanipy.exceptions import (
    InvalidDatasetError,
    InvalidTargetError,
    InvalidTaskError,
    InvalidConfigError,
)

# Internal check imports
from sanipy._checks.categorical_cardinality import check_categorical_cardinality
from sanipy._checks.constant_features import check_constant_features
from sanipy._checks.correlations import check_correlations
from sanipy._checks.dataset_overview import check_overview
from sanipy._checks.duplicate_rows import check_duplicate_rows
from sanipy._checks.identifier_columns import check_identifier_columns
from sanipy._checks.leakage_risk import check_possible_leakage_risk
from sanipy._checks.missing_values import check_missing_values
from sanipy._checks.outlier_detection import check_outlier_detection
from sanipy._checks.distribution_shape import check_distribution_shape
from sanipy._checks.target_analysis import check_target_analysis


def _safe_run(check_name: str, check_func, config: SanipyConfig, *args, **kwargs):
    """Safely run a check function, catching unexpected errors if fail_fast=False."""
    try:
        return check_func(*args, **kwargs)
    except Exception as e:
        if config.fail_fast:
            raise
        return [DiagnosticIssue(
            id=f"internal-error-{check_name}",
            title=f"Sanipy check '{check_name}' failed safely. Please report this as a bug.",
            severity=SEVERITY_LOW,
            category="internal_error",
            evidence={
                "check_name": check_name,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
            recommendation="Please report this trace as an issue in the Sanipy repository.",
            confidence=CONFIDENCE_HIGH,
        )]


def run_engine_checks(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetReport:
    """Run all Sanipy checks internally and compile a DatasetReport."""
    if not isinstance(df, pd.DataFrame):
        raise InvalidDatasetError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if config is not None and not isinstance(config, SanipyConfig):
        raise InvalidConfigError(
            f"config must be a SanipyConfig instance, got {type(config).__name__}."
        )

    if config is None:
        config = SanipyConfig()

    if target is not None:
        if not isinstance(target, (str, int, float, bool, tuple)):
            raise InvalidTargetError(
                f"target must be a string or column label, got type {type(target).__name__}."
            )

    resolved_task = task
    if resolved_task is not None:
        if not isinstance(resolved_task, str):
            raise InvalidTaskError(
                f"task must be a string, got {type(resolved_task).__name__}."
            )
        normalized = resolved_task.strip().lower()
        if normalized not in {"classification", "regression"}:
            raise InvalidTaskError(
                f"task must be 'classification', 'regression', or None -- got {resolved_task!r}."
            )
        resolved_task = normalized

    all_issues: list[DiagnosticIssue] = []

    # Check for MultiIndex columns
    is_multiindex = isinstance(df.columns, pd.MultiIndex)
    if is_multiindex:
        all_issues.append(DiagnosticIssue(
            id="overview-multiindex",
            title="Dataset has MultiIndex columns. This is not fully supported in alpha.",
            severity=SEVERITY_INFO,
            category="overview",
            recommendation="Consider flattening MultiIndex columns for better check compatibility.",
            confidence=CONFIDENCE_HIGH,
        ))

    # Check for duplicate column names
    if not is_multiindex:
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_cols:
            all_issues.append(DiagnosticIssue(
                id="overview-duplicate-columns",
                title=f"Dataset contains duplicate column names: {', '.join(map(str, dup_cols))}.",
                severity=SEVERITY_INFO,
                category="overview",
                evidence={"duplicate_columns": [str(c) for c in dup_cols]},
                recommendation="Ensure all columns have unique names before training models.",
                confidence=CONFIDENCE_HIGH,
            ))

    # ── Sampling guard ──────────────────────────────────────────────
    sampled = False
    working_df = df
    if len(df) > config.max_rows_for_expensive_checks:
        sampled = True
        working_df = df.sample(
            n=config.max_rows_for_expensive_checks,
            random_state=42,
        )
        all_issues.append(DiagnosticIssue(
            id="performance-sampled",
            title=(
                f"Dataset has {len(df):,} rows. Some checks used a "
                f"{config.max_rows_for_expensive_checks:,}-row sample "
                f"for performance."
            ),
            severity=SEVERITY_INFO,
            category=CATEGORY_PERFORMANCE,
            evidence={
                "original_rows": len(df),
                "sampled_rows": config.max_rows_for_expensive_checks,
            },
            recommendation=(
                "Results are approximate. Increase "
                "max_rows_for_expensive_checks in SanipyConfig for "
                "full coverage."
            ),
            confidence=CONFIDENCE_HIGH,
        ))

    # ── 1. Overview (always on full df for accurate counts) ─────
    try:
        overview_issues, dataset_info = check_overview(df, target, config)
        all_issues.extend(overview_issues)
    except Exception as e:
        if config.fail_fast:
            raise
        overview_issues = [DiagnosticIssue(
            id="internal-error-dataset_overview",
            title="Sanipy check 'dataset_overview' failed safely. Please report this as a bug.",
            severity=SEVERITY_LOW,
            category="internal_error",
            evidence={
                "check_name": "dataset_overview",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
            recommendation="Please report this trace as an issue in the Sanipy repository.",
            confidence=CONFIDENCE_HIGH,
        )]
        all_issues.extend(overview_issues)
        dataset_info = {
            "rows": len(df),
            "columns": len(df.columns) if not is_multiindex else len(df.columns.values),
            "memory_mb": 0.0,
            "column_types": {},
        }

    # Early exit if dataset is empty
    if df.empty:
        return DatasetReport(
            issues=all_issues,
            dataset_info=dataset_info,
            task=resolved_task,
            target=target,
        )

    target_missing = target is not None and target not in df.columns

    # ── 2. Missing values (full df for accurate counts) ─────────
    all_issues.extend(_safe_run("missing_values", check_missing_values, config, df, config))

    # ── 3. Duplicates (full df) ─────────────────────────────────
    all_issues.extend(_safe_run("duplicate_rows", check_duplicate_rows, config, df, config))

    # ── 4. Constant columns (full df) ──────────────────────────
    all_issues.extend(_safe_run("constant_features", check_constant_features, config, df, config))

    # ── 5. ID-like columns (full df) ────────────────────────────
    all_issues.extend(_safe_run("identifier_columns", check_identifier_columns, config, df, config, target=target))

    # ── 6. High-cardinality (full df) ──────────────────────────
    all_issues.extend(_safe_run("categorical_cardinality", check_categorical_cardinality, config, df, config, target=target))

    # ── 7. Target checks ────────────────────────────────────────
    if not target_missing:
        try:
            target_issues, resolved_task = check_target_analysis(
                df, target, task, config,
            )
            all_issues.extend(target_issues)
        except Exception as e:
            if config.fail_fast:
                raise
            all_issues.append(DiagnosticIssue(
                id="internal-error-target_analysis",
                title="Sanipy check 'target_analysis' failed safely. Please report this as a bug.",
                severity=SEVERITY_LOW,
                category="internal_error",
                evidence={
                    "check_name": "target_analysis",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                },
                recommendation="Please report this trace as an issue in the Sanipy repository.",
                confidence=CONFIDENCE_HIGH,
            ))

    # ── 8. Outliers (sampled if large) ──────────────────────────
    all_issues.extend(_safe_run("outlier_detection", check_outlier_detection, config, working_df, config, target=target))

    # ── 9. Skewness (sampled if large) ──────────────────────────
    all_issues.extend(_safe_run("distribution_shape", check_distribution_shape, config, working_df, config, target=target))

    # ── 10. Correlation (sampled if large) ──────────────────────
    all_issues.extend(
        _safe_run("correlations", check_correlations, config, working_df, config, target=target, task=resolved_task)
    )

    # ── 11. Leakage heuristics (sampled if large) ───────────────
    if not target_missing:
        all_issues.extend(
            _safe_run("leakage_risk", check_possible_leakage_risk, config, working_df, config, target=target)
        )

    return DatasetReport(
        issues=all_issues,
        dataset_info=dataset_info,
        task=resolved_task,
        target=target,
    )
