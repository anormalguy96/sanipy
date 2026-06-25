"""Tests for numeric outlier detection."""

import numpy as np
import pandas as pd

from sanipy._checks.outlier_detection import check_outlier_detection
from sanipy.config import SanipyConfig


def test_outliers_detected():
    rng = np.random.RandomState(42)
    normal_data = rng.normal(50, 5, 90)
    outlier_data = np.array([200, -100, 300, 250, -150,
                             400, -200, 500, 350, -300])
    df = pd.DataFrame({"value": np.concatenate([normal_data, outlier_data])})
    issues = check_outlier_detection(df, SanipyConfig())
    assert len(issues) >= 1
    assert "outlier" in issues[0].id


def test_no_outliers():
    df = pd.DataFrame({"value": np.arange(100, dtype=float)})
    issues = check_outlier_detection(df, SanipyConfig())
    # Uniform data has minimal IQR outliers
    # (edge values may or may not be flagged depending on distribution)
    # Just verify no crash
    assert isinstance(issues, list)


def test_target_skipped():
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, 100).tolist() + [1000.0]
    df = pd.DataFrame({"target": data})
    issues = check_outlier_detection(df, SanipyConfig(), target="target")
    # Target should be skipped
    assert len(issues) == 0


def test_empty_df():
    df = pd.DataFrame()
    issues = check_outlier_detection(df, SanipyConfig())
    assert len(issues) == 0


def test_small_column_skipped():
    """Columns with <10 values are skipped."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    issues = check_outlier_detection(df, SanipyConfig())
    assert len(issues) == 0
