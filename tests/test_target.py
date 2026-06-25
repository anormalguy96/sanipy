"""Tests for target sanity checks."""

import numpy as np
import pandas as pd

from sanipy._checks.target_analysis import check_target_analysis
from sanipy.config import SanipyConfig


def test_classification_imbalance():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 90 + [1] * 10,
    })
    issues, task = check_target_analysis(df, "target", "classification", SanipyConfig())
    # Should detect imbalance
    assert any("imbalance" in i.id for i in issues)


def test_classification_severe_imbalance():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 98 + [1] * 2,
    })
    issues, task = check_target_analysis(df, "target", "classification", SanipyConfig())
    assert any(i.severity == "critical" and "imbalance" in i.id for i in issues)


def test_classification_balanced():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0] * 50 + [1] * 50,
    })
    issues, task = check_target_analysis(df, "target", "classification", SanipyConfig())
    assert not any("imbalance" in i.id for i in issues)


def test_regression_skewed_target():
    rng = np.random.RandomState(42)
    # Lognormal with high sigma reliably produces skewness > 2
    values = rng.lognormal(0, 2, 1000)
    df = pd.DataFrame({"feature": range(1000), "target": values})
    issues, task = check_target_analysis(df, "target", "regression", SanipyConfig())
    assert any("skewness" in i.id for i in issues)


def test_auto_detect_classification():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0, 1, 2] * 33 + [0],
    })
    issues, task = check_target_analysis(df, "target", None, SanipyConfig())
    assert task == "classification"


def test_auto_detect_regression():
    df = pd.DataFrame({
        "feature": range(100),
        "target": np.arange(100, dtype=float),
    })
    issues, task = check_target_analysis(df, "target", None, SanipyConfig())
    assert task == "regression"


def test_missing_target_values():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [1.0] * 90 + [np.nan] * 10,
    })
    issues, _ = check_target_analysis(df, "target", "regression", SanipyConfig())
    assert any(i.id == "target-missing" for i in issues)


def test_single_class():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [1] * 100,
    })
    issues, _ = check_target_analysis(df, "target", "classification", SanipyConfig())
    assert any(i.id == "target-single-class" for i in issues)


def test_target_not_in_df():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues, task = check_target_analysis(df, "nonexistent", None, SanipyConfig())
    assert len(issues) == 0
    assert task is None
