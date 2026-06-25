"""Sanipy — Lightweight sanity checks for ML datasets.

Quick start::

    from sanipy import check_dataset

    report = check_dataset(df, target="churn")
    print(report.summary())
"""

from __future__ import annotations

from sanipy.api import check_dataset, scan_dataset
from sanipy.comparison import compare_train_test, compare_datasets
from sanipy.comparison_reports import DatasetComparisonReport
from sanipy.config import SanipyConfig
from sanipy.diagnostics import DiagnosticIssue, Issue, Severity
from sanipy.reports import DatasetReport, SanipyReport
from sanipy.exceptions import (
    SanipyError,
    InvalidDatasetError,
    InvalidTargetError,
    InvalidTaskError,
    InvalidConfigError,
    ReportExportError,
)

__version__ = "0.1.0a3"
__all__ = [
    "check_dataset",
    "scan_dataset",
    "compare_train_test",
    "compare_datasets",
    "SanipyConfig",
    "DatasetReport",
    "DatasetComparisonReport",
    "SanipyReport",
    "DiagnosticIssue",
    "Issue",
    "Severity",
    "SanipyError",
    "InvalidDatasetError",
    "InvalidTargetError",
    "InvalidTaskError",
    "InvalidConfigError",
    "ReportExportError",
    "__version__",
]
