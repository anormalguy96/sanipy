"""Tests for correlation checks."""

import numpy as np
import pandas as pd

from sanipy.checks.correlation import check_correlation
from sanipy.config import SanipyConfig


def test_high_correlation_detected():
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, 200)
    df = pd.DataFrame({
        "a": x,
        "b": x + rng.normal(0, 0.01, 200),  # Nearly identical
        "c": rng.normal(0, 1, 200),  # Independent
    })
    issues = check_correlation(df, SanipyConfig())
    assert any("correlation-a-b" == i.id for i in issues)


def test_no_correlation():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(0, 1, 200),
    })
    issues = check_correlation(df, SanipyConfig())
    # Independent columns should not be flagged
    corr_issues = [i for i in issues if i.category == "correlation"]
    assert len(corr_issues) == 0


def test_too_many_columns_skipped():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        rng.normal(0, 1, (100, 150)),
        columns=[f"col_{i}" for i in range(150)],
    )
    config = SanipyConfig(max_columns_for_correlation=100)
    issues = check_correlation(df, config)
    assert any(i.id == "correlation-skipped" for i in issues)


def test_single_column():
    df = pd.DataFrame({"a": range(100)})
    issues = check_correlation(df, SanipyConfig())
    assert len(issues) == 0


def test_empty_df():
    df = pd.DataFrame()
    issues = check_correlation(df, SanipyConfig())
    assert len(issues) == 0
