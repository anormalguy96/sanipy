"""Tests for missing value detection."""

import numpy as np
import pandas as pd

from sanipy.checks.missing import check_missing_values
from sanipy.config import SanipyConfig


def test_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 0


def test_small_missing():
    # 3/100 = 3% which is below missing_low_threshold (5%), so severity = info
    df = pd.DataFrame({"a": [1, 2, np.nan] + list(range(97))})
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].severity == "info"


def test_low_missing():
    # 8/100 = 8% which is above missing_low_threshold (5%), so severity = low
    data = [np.nan] * 8 + list(range(92))
    df = pd.DataFrame({"a": data})
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].severity == "low"


def test_high_missing():
    data = [np.nan] * 50 + list(range(50))
    df = pd.DataFrame({"a": data})
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].severity == "high"


def test_critical_missing():
    data = [np.nan] * 80 + list(range(20))
    df = pd.DataFrame({"a": data})
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 1
    assert issues[0].severity == "critical"


def test_empty_df():
    df = pd.DataFrame()
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 0


def test_multiple_columns_missing():
    df = pd.DataFrame({
        "a": [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "b": [np.nan] * 10 + list(range(10)),
    })
    issues = check_missing_values(df, SanipyConfig())
    assert len(issues) == 2
