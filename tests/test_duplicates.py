"""Tests for duplicate row detection."""

import pandas as pd

from sanipy.checks.duplicates import check_duplicates
from sanipy.config import SanipyConfig


def test_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    issues = check_duplicates(df, SanipyConfig())
    assert len(issues) == 0


def test_some_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 3] * 25})
    issues = check_duplicates(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].evidence["duplicate_count"] > 0


def test_all_duplicates():
    df = pd.DataFrame({"a": [1] * 100})
    issues = check_duplicates(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].severity in ("medium", "high")


def test_empty_df():
    df = pd.DataFrame()
    issues = check_duplicates(df, SanipyConfig())
    assert len(issues) == 0
