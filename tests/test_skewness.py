"""Tests for skewness detection."""

import numpy as np
import pandas as pd

from sanipy._checks.distribution_shape import check_distribution_shape
from sanipy.config import SanipyConfig


def test_skewed_column():
    rng = np.random.RandomState(42)
    # Lognormal with high sigma produces reliably high skewness (>2)
    df = pd.DataFrame({"a": rng.lognormal(0, 2, 1000)})
    issues = check_distribution_shape(df, SanipyConfig())
    assert len(issues) >= 1
    assert "skewness" in issues[0].id


def test_normal_column():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"a": rng.normal(0, 1, 1000)})
    issues = check_distribution_shape(df, SanipyConfig())
    # Normal distribution skewness is ~0, should not trigger
    assert len(issues) == 0


def test_target_skipped():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"target": rng.exponential(10, 1000)})
    issues = check_distribution_shape(df, SanipyConfig(), target="target")
    assert len(issues) == 0


def test_empty_df():
    df = pd.DataFrame()
    issues = check_distribution_shape(df, SanipyConfig())
    assert len(issues) == 0
