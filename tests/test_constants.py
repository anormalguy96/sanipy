"""Tests for constant / near-constant column detection."""

import pandas as pd

from sanipy.checks.constants import check_constant_columns
from sanipy.config import SanipyConfig


def test_constant_column():
    df = pd.DataFrame({"a": [42] * 100, "b": range(100)})
    issues = check_constant_columns(df, SanipyConfig())
    ids = [i.id for i in issues]
    assert "constant-a" in ids
    assert "constant-b" not in ids


def test_near_constant_column():
    df = pd.DataFrame({"a": [1] * 99 + [2]})
    issues = check_constant_columns(df, SanipyConfig())
    ids = [i.id for i in issues]
    assert "near-constant-a" in ids


def test_no_constant_columns():
    df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    issues = check_constant_columns(df, SanipyConfig())
    assert len(issues) == 0


def test_all_null_column():
    df = pd.DataFrame({"a": [None] * 100})
    issues = check_constant_columns(df, SanipyConfig())
    # All null → 0 unique values → constant
    assert any("constant" in i.id for i in issues)


def test_empty_df():
    df = pd.DataFrame()
    issues = check_constant_columns(df, SanipyConfig())
    assert len(issues) == 0
