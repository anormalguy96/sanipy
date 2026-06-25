"""Public Train/Test comparison API for Sanipy."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.comparison_reports import DatasetComparisonReport
from sanipy._comparison.runner import run_comparison_checks


def compare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetComparisonReport:
    """Compare training and test datasets to detect split inconsistencies and leakage.

    Parameters:
        train_df: Training dataset. **Not mutated.**
        test_df: Test dataset. **Not mutated.**
        target: Name of the target/label column (optional).
        task: ``"classification"``, ``"regression"``, or ``None``
            (auto-detected from target if possible).
        config: Optional :class:`SanipyConfig` to override default thresholds.

    Returns:
        A :class:`DatasetComparisonReport` containing all detected issues.
    """
    return run_comparison_checks(
        train_df=train_df,
        test_df=test_df,
        target=target,
        task=task,
        config=config,
    )


# Alias
compare_datasets = compare_train_test
