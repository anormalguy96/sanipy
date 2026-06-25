"""Core orchestrator — ``check_dataset()`` entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sanipy.checks.cardinality import check_high_cardinality
from sanipy.checks.constants import check_constant_columns
from sanipy.checks.correlation import check_correlation
from sanipy.checks.duplicates import check_duplicates
from sanipy.checks.id_columns import check_id_columns
from sanipy.checks.leakage import check_leakage
from sanipy.checks.missing import check_missing_values
from sanipy.checks.outliers import check_outliers
from sanipy.checks.overview import check_overview
from sanipy.checks.skewness import check_skewness
from sanipy.checks.target import check_target
from sanipy.config import SanipyConfig
from sanipy.issues import (
    CATEGORY_PERFORMANCE,
    CONFIDENCE_HIGH,
    SEVERITY_INFO,
    Issue,
)
from sanipy.report import SanipyReport

if TYPE_CHECKING:
    pass


def check_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> SanipyReport:
    """Run all Sanipy sanity checks on a pandas DataFrame.

    Parameters:
        df: The dataset to check. **Not mutated.**
        target: Name of the target/label column (optional).
        task: ``"classification"``, ``"regression"``, or ``None``
            (auto-detected from target if possible).
        config: Optional :class:`SanipyConfig` to override default
            thresholds.

    Returns:
        A :class:`SanipyReport` with all detected issues, a heuristic
        health score, and export capabilities.

    Example::

        from sanipy import check_dataset

        report = check_dataset(df, target="churn")
        print(report.summary())
    """
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

    all_issues: list[Issue] = []
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
        all_issues.append(Issue(
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
        return SanipyReport(
            issues=all_issues,
            dataset_info=dataset_info,
            task=resolved_task,
            target=target,
        )

    target_missing = target is not None and target not in df.columns
    if target_missing:
        # Still run non-target checks
        pass

    # ── 2. Missing values (full df for accurate counts) ─────────
    all_issues.extend(check_missing_values(df, config))

    # ── 3. Duplicates (full df) ─────────────────────────────────
    all_issues.extend(check_duplicates(df, config))

    # ── 4. Constant columns (full df) ──────────────────────────
    all_issues.extend(check_constant_columns(df, config))

    # ── 5. ID-like columns (full df) ────────────────────────────
    all_issues.extend(check_id_columns(df, config, target=target))

    # ── 6. High-cardinality (full df) ──────────────────────────
    all_issues.extend(check_high_cardinality(df, config, target=target))

    # ── 7. Target checks ────────────────────────────────────────
    if not target_missing:
        target_issues, resolved_task = check_target(
            df, target, task, config,
        )
        all_issues.extend(target_issues)

    # ── 8. Outliers (sampled if large) ──────────────────────────
    all_issues.extend(check_outliers(working_df, config, target=target))

    # ── 9. Skewness (sampled if large) ──────────────────────────
    all_issues.extend(check_skewness(working_df, config, target=target))

    # ── 10. Correlation (sampled if large) ──────────────────────
    all_issues.extend(
        check_correlation(working_df, config, target=target, task=resolved_task)
    )

    # ── 11. Leakage heuristics (sampled if large) ───────────────
    if not target_missing:
        all_issues.extend(
            check_leakage(working_df, config, target=target)
        )

    return SanipyReport(
        issues=all_issues,
        dataset_info=dataset_info,
        task=resolved_task,
        target=target,
    )
