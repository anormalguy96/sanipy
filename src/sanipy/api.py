"""Public entrypoints for Sanipy: check_dataset and scan_dataset."""

from __future__ import annotations

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy._engine.runner import run_engine_checks
from sanipy.reports import DatasetReport
from sanipy.comparison import compare_train_test, compare_datasets


def check_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetReport:
    """Run all Sanipy dataset checks and compile a report.

    Parameters:
        df: The dataset to check. **Not mutated.**
        target: Name of the target/label column (optional).
        task: ``"classification"``, ``"regression"``, or ``None``
            (auto-detected from target if possible).
        config: Optional :class:`SanipyConfig` to override default
            thresholds.

    Returns:
        A :class:`DatasetReport` containing all detected issues.
    """
    return run_engine_checks(df=df, target=target, task=task, config=config)


def scan_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetReport:
    """Run all Sanipy dataset checks and compile a report (alias of check_dataset).

    Parameters:
        df: The dataset to check. **Not mutated.**
        target: Name of the target/label column (optional).
        task: ``"classification"``, ``"regression"``, or ``None``
            (auto-detected from target if possible).
        config: Optional :class:`SanipyConfig` to override default
            thresholds.

    Returns:
        A :class:`DatasetReport` containing all detected issues.
    """
    return run_engine_checks(df=df, target=target, task=task, config=config)

