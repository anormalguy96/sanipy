"""Sanipy — Lightweight sanity checks for ML datasets.

Quick start::

    from sanipy import check_dataset

    report = check_dataset(df, target="churn")
    print(report.summary())
"""

from __future__ import annotations

from sanipy.config import SanipyConfig
from sanipy.core import check_dataset
from sanipy.issues import Issue
from sanipy.report import SanipyReport

__version__ = "0.1.0"
__all__ = [
    "check_dataset",
    "SanipyConfig",
    "SanipyReport",
    "Issue",
    "__version__",
]
