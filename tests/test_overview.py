"""Tests for overview checks."""

import pandas as pd

from sanipy._checks.dataset_overview import check_overview
from sanipy.config import SanipyConfig


def test_empty_dataset():
    df = pd.DataFrame()
    config = SanipyConfig()
    issues, info = check_overview(df, target=None, config=config)
    assert any(i.id == "overview-001" for i in issues)
    assert info["rows"] == 0


def test_small_dataset():
    df = pd.DataFrame({"a": range(10)})
    config = SanipyConfig()
    issues, info = check_overview(df, target=None, config=config)
    assert any(i.id == "overview-002" for i in issues)


def test_missing_target_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    config = SanipyConfig()
    issues, info = check_overview(df, target="nonexistent", config=config)
    assert any(i.id == "overview-003" for i in issues)


def test_normal_dataset():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    config = SanipyConfig()
    issues, info = check_overview(df, target=None, config=config)
    assert info["rows"] == 100
    assert info["columns"] == 2
    # No critical issues expected
    assert not any(i.severity == "critical" for i in issues)
