"""Private orchestration engine for running Sanipy dataset checks."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_PERFORMANCE,
    CONFIDENCE_HIGH,
    SEVERITY_INFO,
    DiagnosticIssue,
)
from sanipy.reports import DatasetReport

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


def run_engine_checks(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetReport:
    """Run all Sanipy checks internally and compile a DatasetReport."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    if config is None:
        config = SanipyConfig()

    # Validate task parameter
    valid_tasks = {"classification", "regression", None}
    if task not in valid_tasks:
        raise ValueError(
            f"task must be 'classification', 'regression', or None -- "
            f"got {task!r}."
        )

    all_issues: list[DiagnosticIssue] = []
    resolved_task = task

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
    overview_issues, dataset_info = check_overview(df, target, config)
    all_issues.extend(overview_issues)

    # Early exit if dataset is empty or target not found
    if df.empty:
        return DatasetReport(
            issues=all_issues,
            dataset_info=dataset_info,
            task=resolved_task,
            target=target,
        )

    target_missing = target is not None and target not in df.columns

    # ── 2. Missing values (full df for accurate counts) ─────────
    all_issues.extend(check_missing_values(df, config))

    # ── 3. Duplicates (full df) ─────────────────────────────────
    all_issues.extend(check_duplicate_rows(df, config))

    # ── 4. Constant columns (full df) ──────────────────────────
    all_issues.extend(check_constant_features(df, config))

    # ── 5. ID-like columns (full df) ────────────────────────────
    all_issues.extend(check_identifier_columns(df, config, target=target))

    # ── 6. High-cardinality (full df) ──────────────────────────
    all_issues.extend(check_categorical_cardinality(df, config, target=target))

    # ── 7. Target checks ────────────────────────────────────────
    if not target_missing:
        target_issues, resolved_task = check_target_analysis(
            df, target, task, config,
        )
        all_issues.extend(target_issues)

    # ── 8. Outliers (sampled if large) ──────────────────────────
    all_issues.extend(check_outlier_detection(working_df, config, target=target))

    # ── 9. Skewness (sampled if large) ──────────────────────────
    all_issues.extend(check_distribution_shape(working_df, config, target=target))

    # ── 10. Correlation (sampled if large) ──────────────────────
    all_issues.extend(
        check_correlations(working_df, config, target=target, task=resolved_task)
    )

    # ── 11. Leakage heuristics (sampled if large) ───────────────
    if not target_missing:
        all_issues.extend(
            check_possible_leakage_risk(working_df, config, target=target)
        )

    return DatasetReport(
        issues=all_issues,
        dataset_info=dataset_info,
        task=resolved_task,
        target=target,
    )
