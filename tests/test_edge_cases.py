"""Edge case tests for Sanipy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from sanipy import (
    check_dataset,
    SanipyConfig,
    DatasetReport,
    InvalidDatasetError,
    InvalidTargetError,
    InvalidTaskError,
    InvalidConfigError,
    ReportExportError,
)


# 1. Non-DataFrame input
def test_non_dataframe_input():
    with pytest.raises(InvalidDatasetError):
        check_dataset("not a dataframe")  # type: ignore


# 2. Empty DataFrame
# 3. DataFrame with no columns
# 4. DataFrame with no rows
def test_empty_dataframe_variations():
    # empty
    df_empty = pd.DataFrame()
    report = check_dataset(df_empty)
    assert len(report.issues) > 0
    assert any(i.id == "overview-001" for i in report.issues)  # empty rows
    assert any(i.id == "overview-004" for i in report.issues)  # empty columns

    # no columns
    df_no_cols = pd.DataFrame(index=[1, 2, 3])
    report = check_dataset(df_no_cols)
    assert any(i.id == "overview-004" for i in report.issues)

    # no rows but has columns
    df_no_rows = pd.DataFrame(columns=["a", "b"])
    report = check_dataset(df_no_rows)
    assert any(i.id == "overview-001" for i in report.issues)


# 5. Missing target
def test_missing_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = check_dataset(df, target="target")
    assert any("not found in the dataset" in i.title for i in report.issues)


# 6. Invalid task
def test_invalid_task():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(InvalidTaskError):
        check_dataset(df, task="invalid_task")


# 7. Invalid config object
def test_invalid_config_object():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(InvalidConfigError):
        check_dataset(df, config="not a config"  # type: ignore
                      )

    # config constraints
    with pytest.raises(InvalidConfigError):
        SanipyConfig(missing_low_threshold=-0.1)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(missing_low_threshold=1.5)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(missing_low_threshold=0.5, missing_medium_threshold=0.4)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(high_cardinality_threshold=0)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(outlier_iqr_multiplier=-1.0)


# 8. Duplicate column names
def test_duplicate_column_names():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    report = check_dataset(df, target="b")
    # Duplicate column issue should be present
    assert any(i.id == "overview-duplicate-columns" for i in report.issues)
    # The check run itself should not crash
    assert report.score >= 0


# 9. Non-string column names
def test_non_string_column_names():
    df = pd.DataFrame({
        1: [1, 2, 3],
        2.5: [4, 5, 6],
        True: [7, 8, 9],
        "target": [0, 1, 0]
    })
    # Should not crash on name-based heuristics
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 10. MultiIndex columns
def test_multiindex_columns():
    cols = pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")])
    df = pd.DataFrame([[1, 2], [3, 4]], columns=cols)
    report = check_dataset(df)
    assert any(i.id == "overview-multiindex" for i in report.issues)


# 11. All-null column
def test_all_null_column():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [None, None, None],
        "target": [0, 1, 0]
    })
    report = check_dataset(df, target="target")
    # Column b should be constant/all-null
    assert any("constant" in i.id for i in report.issues)


# 12. All-null target
def test_all_null_target():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "target": [None, None, None]
    })
    report = check_dataset(df, target="target")
    assert any(i.id == "target-all-null" for i in report.issues)


# 13. Single-class classification target
def test_single_class_target():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "target": [1, 1, 1]
    })
    report = check_dataset(df, target="target", task="classification")
    assert any(i.id == "target-single-class" for i in report.issues)


# 14. Constant regression target
def test_constant_regression_target():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "target": [5.0, 5.0, 5.0]
    })
    report = check_dataset(df, target="target", task="regression")
    assert any(i.id == "target-constant-regression" for i in report.issues)


# 15. No numeric columns
def test_no_numeric_columns():
    df = pd.DataFrame({
        "a": ["x", "y", "z"],
        "target": ["cat", "dog", "cat"]
    })
    report = check_dataset(df, target="target", task="classification")
    assert report.score >= 0


# 16. No categorical columns
def test_no_categorical_columns():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "target": [4.0, 5.0, 6.0]
    })
    report = check_dataset(df, target="target", task="regression")
    assert report.score >= 0


# 17. Datetime columns
def test_datetime_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "val": range(10),
        "target": [0, 1] * 5
    })
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 18. Boolean columns
def test_boolean_columns():
    df = pd.DataFrame({
        "flag": [True, False, True, True, False, False, True, False, True, False],
        "target": [0, 1] * 5
    })
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 19. Nullable integer/string/boolean pandas dtypes
def test_nullable_pandas_dtypes():
    df = pd.DataFrame({
        "nullable_int": pd.Series([1, None, 3, 4, 5, 6, 7, 8, 9, 10], dtype="Int64"),
        "nullable_str": pd.Series(["a", None, "c", "d", "e", "f", "g", "h", "i", "j"], dtype="string"),
        "nullable_bool": pd.Series([True, None, False, True, False, True, False, True, False, True], dtype="boolean"),
        "target": [0, 1] * 5
    })
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 20. Infinite numeric values
def test_infinite_numeric_values():
    df = pd.DataFrame({
        "inf_col": [1.0, 2.0, np.inf, 4.0, -np.inf, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1] * 5
    })
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 21. Mixed object columns
def test_mixed_object_columns():
    df = pd.DataFrame({
        "mixed": [1, "two", 3.0, None, 5, "six", 7, 8, 9, 10],
        "target": [0, 1] * 5
    })
    report = check_dataset(df, target="target")
    assert report.score >= 0


# 22. Very small dataset
def test_very_small_dataset():
    df = pd.DataFrame({
        "a": [1, 2],
        "target": [0, 1]
    })
    report = check_dataset(df, target="target")
    assert any(i.id == "overview-002" for i in report.issues)


# 23. Wide dataset triggering correlation guard
def test_wide_dataset_correlation_guard():
    data = {f"col_{i}": range(10) for i in range(105)}
    df = pd.DataFrame(data)
    config = SanipyConfig(max_columns_for_correlation=100)
    report = check_dataset(df, config=config)
    assert any(i.id == "correlation-skipped" for i in report.issues)


# 24. Large dataset triggering sampling guard
def test_large_dataset_sampling_guard():
    df = pd.DataFrame({
        "a": range(1000),
        "target": [0, 1] * 500
    })
    config = SanipyConfig(max_rows_for_expensive_checks=100)
    report = check_dataset(df, target="target", config=config)
    assert any(i.id == "performance-sampled" for i in report.issues)


# 25. Input DataFrame is not mutated
def test_dataframe_not_mutated():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, None, 6]
    })
    df_orig = df.copy()
    check_dataset(df)
    pd.testing.assert_frame_equal(df, df_orig)


# fail_fast behaviors
def test_fail_fast_behavior():
    df = pd.DataFrame({"a": [1, 2, 3]})
    config_fail_fast = SanipyConfig(fail_fast=True)
    config_safe = SanipyConfig(fail_fast=False)

    from sanipy._engine.runner import _safe_run

    def buggy_check(*args, **kwargs):
        raise RuntimeError("Something went wrong")

    issues = _safe_run("buggy_check", buggy_check, config_safe)
    assert len(issues) == 1
    assert issues[0].category == "internal_error"
    assert "buggy_check" in issues[0].evidence["check_name"]

    with pytest.raises(RuntimeError):
        _safe_run("buggy_check", buggy_check, config_fail_fast)


# Export robustness and exceptions
def test_export_robustness(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, np.inf, np.nan],
        "b": [pd.NA, "hello", pd.Timestamp("2026-06-25"), 42],
        "my|target": [0, 1, 0, 1]
    })
    report = check_dataset(df, target="my|target")

    d = report.to_dict()
    assert d["dataset_info"]["rows"] == 4

    js = report.to_json()
    assert "Infinity" in js or "null" in js

    md = report.to_markdown()
    assert "my\\|target" in md

    with pytest.raises(ReportExportError):
        report.save(tmp_path / "report.invalid")

    with pytest.raises(ReportExportError):
        report.save(tmp_path / "non_existent_folder" / "report.json")
