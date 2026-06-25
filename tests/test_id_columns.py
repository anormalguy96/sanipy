"""Tests for ID-like column detection."""

import pandas as pd

from sanipy.checks.id_columns import check_id_columns
from sanipy.config import SanipyConfig


def test_column_named_id():
    df = pd.DataFrame({"id": range(100), "value": range(100)})
    issues = check_id_columns(df, SanipyConfig())
    assert any("id-column-id" == i.id for i in issues)


def test_column_named_customer_id():
    df = pd.DataFrame({"customer_id": range(100), "value": range(100)})
    issues = check_id_columns(df, SanipyConfig())
    assert len(issues) >= 1


def test_high_uniqueness_string_column():
    df = pd.DataFrame({
        "code": [f"item_{i}" for i in range(100)],
        "value": range(100),
    })
    issues = check_id_columns(df, SanipyConfig())
    assert len(issues) >= 1


def test_float_column_not_flagged():
    """Floats with high uniqueness are continuous features, not IDs."""
    import numpy as np
    df = pd.DataFrame({
        "measurement": np.random.random(100),
    })
    issues = check_id_columns(df, SanipyConfig())
    assert len(issues) == 0


def test_target_not_flagged():
    df = pd.DataFrame({"id": range(100)})
    issues = check_id_columns(df, SanipyConfig(), target="id")
    assert len(issues) == 0


def test_empty_df():
    df = pd.DataFrame()
    issues = check_id_columns(df, SanipyConfig())
    assert len(issues) == 0
