"""Sanipy — Lightweight sanity checks for ML datasets.

Quick start::

    from sanipy import check_dataset

    report = check_dataset(df, target="churn")
    print(report.summary())
"""

from __future__ import annotations

from sanipy.api import check_dataset, scan_dataset
from sanipy.config import SanipyConfig
from sanipy.diagnostics import DiagnosticIssue, Issue, Severity
from sanipy.reports import DatasetReport, SanipyReport

__version__ = "0.1.0"
__all__ = [
    "check_dataset",
    "scan_dataset",
    "SanipyConfig",
    "DatasetReport",
    "SanipyReport",
    "DiagnosticIssue",
    "Issue",
    "Severity",
    "__version__",
]
