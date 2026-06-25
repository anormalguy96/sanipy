"""Integration tests for check_dataset() end-to-end."""

import numpy as np
import pandas as pd
import pytest

from sanipy import check_dataset, SanipyConfig, SanipyReport


def test_basic_usage():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "customer_id": range(500),
        "age": rng.normal(35, 10, 500),
        "income": rng.exponential(50000, 500),
        "category": rng.choice(["A", "B", "C"], 500),
        "churn": rng.choice([0, 1], 500, p=[0.85, 0.15]),
    })
    report = check_dataset(df, target="churn")
    assert isinstance(report, SanipyReport)
    assert 0 <= report.score <= 100
    assert len(report) > 0

    # Should detect ID column
    assert any("id-column" in i.id for i in report.issues)

    # Should detect imbalance
    assert any("imbalance" in i.id for i in report.issues)

    # Summary should be a string
    text = report.summary()
    assert "Sanipy" in text
    assert "churn" in text


def test_regression_task():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, 300),
        "feature_b": rng.normal(0, 1, 300),
        "price": rng.exponential(100, 300),
    })
    report = check_dataset(df, target="price", task="regression")
    assert report.task == "regression"
    assert isinstance(report.summary(), str)


def test_no_target():
    df = pd.DataFrame({
        "a": range(100),
        "b": range(100),
    })
    report = check_dataset(df)
    assert isinstance(report, SanipyReport)


def test_empty_dataframe():
    df = pd.DataFrame()
    report = check_dataset(df)
    assert report.score < 100
    assert any(i.severity == "critical" for i in report.issues)


def test_custom_config():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "feature": rng.normal(0, 1, 100),
        "target": rng.choice([0, 1], 100, p=[0.7, 0.3]),
    })
    config = SanipyConfig(imbalance_majority_threshold=0.6)
    report = check_dataset(df, target="target", config=config)
    assert any("imbalance" in i.id for i in report.issues)


def test_invalid_input():
    with pytest.raises(TypeError):
        check_dataset("not a dataframe")  # type: ignore[arg-type]


def test_invalid_task():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        check_dataset(df, task="unsupervised")


def test_dataframe_not_mutated():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": [np.nan] * 30 + list(range(70)),
        "target": rng.choice([0, 1], 100),
    })
    original_shape = df.shape
    original_values = df.values.copy()

    check_dataset(df, target="target")

    assert df.shape == original_shape
    # Compare non-NaN values
    mask = ~pd.isna(df.values) & ~pd.isna(original_values)
    assert (df.values[mask] == original_values[mask]).all()


def test_export_roundtrip(tmp_path):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "target": rng.choice([0, 1], 100),
    })
    report = check_dataset(df, target="target")

    # JSON
    json_path = tmp_path / "report.json"
    report.save(json_path)
    assert json_path.exists()

    # Markdown
    md_path = tmp_path / "report.md"
    report.save(md_path)
    assert md_path.exists()

    # Text
    txt_path = tmp_path / "report.txt"
    report.save(txt_path)
    assert txt_path.exists()
