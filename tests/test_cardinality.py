"""Tests for high-cardinality categorical detection."""

import pandas as pd

from sanipy.checks.cardinality import check_high_cardinality
from sanipy.config import SanipyConfig


def test_high_cardinality():
    df = pd.DataFrame({
        "city": [f"city_{i}" for i in range(200)],
        "value": range(200),
    })
    issues = check_high_cardinality(df, SanipyConfig())
    assert len(issues) == 1
    assert "city" in issues[0].columns


def test_low_cardinality():
    df = pd.DataFrame({
        "color": ["red", "blue", "green"] * 33 + ["red"],
    })
    issues = check_high_cardinality(df, SanipyConfig())
    assert len(issues) == 0


def test_target_not_flagged():
    df = pd.DataFrame({
        "label": [f"class_{i}" for i in range(100)],
    })
    issues = check_high_cardinality(df, SanipyConfig(), target="label")
    assert len(issues) == 0


def test_custom_threshold():
    df = pd.DataFrame({
        "col": [f"val_{i}" for i in range(30)],
    })
    config = SanipyConfig(high_cardinality_threshold=20)
    issues = check_high_cardinality(df, config)
    assert len(issues) == 1


def test_empty_df():
    df = pd.DataFrame()
    issues = check_high_cardinality(df, SanipyConfig())
    assert len(issues) == 0
