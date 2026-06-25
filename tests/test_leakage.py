"""Tests for leakage heuristics."""

import numpy as np
import pandas as pd

from sanipy.checks.leakage import check_leakage
from sanipy.config import SanipyConfig


def test_suspicious_name_and_high_correlation():
    rng = np.random.RandomState(42)
    target = rng.choice([0, 1], 200)
    df = pd.DataFrame({
        "feature": rng.normal(0, 1, 200),
        "final_result": target.astype(float),  # Suspicious name + perfect corr
        "target": target,
    })
    issues = check_leakage(df, SanipyConfig(), target="target")
    assert any("leakage" in i.id for i in issues)


def test_suspicious_name_only():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "resolved_at": rng.normal(0, 1, 200),  # Suspicious name, low correlation
        "target": rng.choice([0, 1], 200),
    })
    issues = check_leakage(df, SanipyConfig(), target="target")
    assert any("leakage-name" in i.id for i in issues)


def test_high_correlation_only():
    target = np.arange(200, dtype=float)
    df = pd.DataFrame({
        "perfect_copy": target + 0.001,
        "target": target,
    })
    issues = check_leakage(df, SanipyConfig(), target="target")
    assert any("leakage" in i.id for i in issues)


def test_no_leakage():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "feature": rng.normal(0, 1, 200),
        "target": rng.choice([0, 1], 200),
    })
    issues = check_leakage(df, SanipyConfig(), target="target")
    assert len(issues) == 0


def test_no_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = check_leakage(df, SanipyConfig(), target=None)
    assert len(issues) == 0


def test_empty_df():
    df = pd.DataFrame()
    issues = check_leakage(df, SanipyConfig(), target="a")
    assert len(issues) == 0
