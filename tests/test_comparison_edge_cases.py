"""Edge cases and error handling tests for Sanipy Train/Test comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sanipy import (
    compare_train_test,
    SanipyConfig,
    DiagnosticIssue,
)
from sanipy.exceptions import InvalidDatasetError, InvalidConfigError


def test_empty_train_or_test():
    """Verify empty DataFrames are handled safely."""
    empty_df = pd.DataFrame()
    valid_df = pd.DataFrame({"col": [1, 2]})

    # Train empty
    report_train_empty = compare_train_test(empty_df, valid_df)
    assert any(i.id == "comparison-overview-empty-train" for i in report_train_empty.issues)

    # Test empty
    report_test_empty = compare_train_test(valid_df, empty_df)
    assert any(i.id == "comparison-overview-empty-test" for i in report_test_empty.issues)


def test_duplicate_columns():
    """Verify duplicate columns are handled without crashing."""
    train_df = pd.DataFrame([[1, 2], [3, 4]], columns=["col", "col"])
    test_df = pd.DataFrame([[5, 6], [7, 8]], columns=["col", "col"])

    report = compare_train_test(train_df, test_df)
    assert any(i.id == "comparison-schema-duplicate-train" for i in report.issues)
    assert any(i.id == "comparison-schema-duplicate-test" for i in report.issues)


def test_non_string_columns():
    """Verify numeric/boolean column labels do not crash comparison checks."""
    train_df = pd.DataFrame({1: [10, 20], True: ["a", "b"]})
    test_df = pd.DataFrame({1: [15, 25], True: ["a", "c"]})

    report = compare_train_test(train_df, test_df)
    assert len(report.issues) > 0  # Should run check successfully without type error crashes


def test_multiindex_columns():
    """Verify MultiIndex columns are handled safely and trigger warning."""
    cols = pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    train_df = pd.DataFrame([[1, 2], [3, 4]], columns=cols)
    test_df = pd.DataFrame([[5, 6], [7, 8]], columns=cols)

    report = compare_train_test(train_df, test_df)
    assert any(i.id == "comparison-schema-multiindex" for i in report.issues)


def test_no_numeric_or_categorical_columns():
    """Verify datasets with only numeric, or only categorical columns, or neither do not crash."""
    # Only numeric
    train_num = pd.DataFrame({"col": [1, 2]})
    test_num = pd.DataFrame({"col": [3, 4]})
    report_num = compare_train_test(train_num, test_num)
    assert isinstance(report_num.score, int)

    # Only categorical
    train_cat = pd.DataFrame({"col": ["a", "b"]})
    test_cat = pd.DataFrame({"col": ["b", "c"]})
    report_cat = compare_train_test(train_cat, test_cat)
    assert isinstance(report_cat.score, int)

    # No columns (handled in empty/no-cols check)
    train_empty = pd.DataFrame(index=[0, 1])
    test_empty = pd.DataFrame(index=[0, 1])
    report_empty = compare_train_test(train_empty, test_empty)
    assert any(i.id == "comparison-overview-no-cols-train" for i in report_empty.issues)


def test_nan_inf_values():
    """Verify NaNs and Infinities do not crash comparisons."""
    train_df = pd.DataFrame({"col": [1.0, np.nan, np.inf, -np.inf]})
    test_df = pd.DataFrame({"col": [np.nan, 2.0, np.inf, 3.0]})

    report = compare_train_test(train_df, test_df)
    assert isinstance(report.score, int)


def test_all_null_columns():
    """Verify all-null columns do not crash the checks."""
    train_df = pd.DataFrame({"col": [np.nan, np.nan]})
    test_df = pd.DataFrame({"col": [np.nan, np.nan]})

    report = compare_train_test(train_df, test_df)
    assert isinstance(report.score, int)


def test_very_small_datasets():
    """Verify very small datasets (e.g. 1 or 2 rows) do not crash."""
    train_df = pd.DataFrame({"col": [1]})
    test_df = pd.DataFrame({"col": [2]})

    report = compare_train_test(train_df, test_df)
    assert isinstance(report.score, int)
    assert any(i.id == "comparison-overview-small-test" for i in report.issues)


def test_large_dataset_overlap_guard():
    """Verify that exact row overlap is skipped if row count exceeds configuration limit."""
    train_df = pd.DataFrame({"col": range(10)})
    test_df = pd.DataFrame({"col": range(10)})

    # Max rows for overlap check is set to 5
    config = SanipyConfig(comparison_max_rows_for_overlap_check=5)
    report = compare_train_test(train_df, test_df, config=config)

    assert any(i.id == "comparison-row-overlap-skipped" for i in report.issues)
    assert not any(i.id == "comparison-row-overlap" for i in report.issues)


def test_dataframe_not_mutated():
    """Verify input DataFrames are not modified during the comparison process."""
    train_df = pd.DataFrame({"feat": [1, 2], "target": ["A", "B"]})
    test_df = pd.DataFrame({"feat": [3, 4], "target": ["A", "C"]})

    train_orig = train_df.copy()
    test_orig = test_df.copy()

    compare_train_test(train_df, test_df, target="target", task="classification")

    pd.testing.assert_frame_equal(train_df, train_orig)
    pd.testing.assert_frame_equal(test_df, test_orig)


def test_fail_fast_behavior():
    """Verify fail_fast behaves correctly with internal check failures."""
    train_df = pd.DataFrame({"col": [1, 2]})
    test_df = pd.DataFrame({"col": [3, 4]})

    # Inject a failing function for schema_mismatch
    # We can temporarily mock check_schema_mismatch
    import sanipy._comparison.runner as runner
    orig_check = runner.check_schema_mismatch

    def failing_check(*args, **kwargs):
        raise ValueError("Simulated check crash")

    runner.check_schema_mismatch = failing_check

    try:
        # fail_fast = False (default): captures error as DiagnosticIssue and continues
        config_no_ff = SanipyConfig(fail_fast=False)
        report = compare_train_test(train_df, test_df, config=config_no_ff)
        assert any(i.id == "internal-error-comparison-schema_mismatch" for i in report.issues)

        # fail_fast = True: raises exception immediately
        config_ff = SanipyConfig(fail_fast=True)
        with pytest.raises(ValueError, match="Simulated check crash"):
            compare_train_test(train_df, test_df, config=config_ff)
    finally:
        # Restore original function
        runner.check_schema_mismatch = orig_check


def test_invalid_input_types():
    """Verify passing non-DataFrame raises InvalidDatasetError."""
    with pytest.raises(InvalidDatasetError, match="Expected training dataset to be a pandas DataFrame"):
        compare_train_test([1, 2], pd.DataFrame())

    with pytest.raises(InvalidDatasetError, match="Expected testing dataset to be a pandas DataFrame"):
        compare_train_test(pd.DataFrame(), [1, 2])


def test_invalid_comparison_configs():
    """Verify that invalid comparison config fields raise InvalidConfigError."""
    # percentage thresholds bounds (0.0 to 1.0)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_missingness_shift_threshold=-0.1)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_missingness_shift_threshold=1.1)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_unseen_category_row_threshold=-0.01)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_unseen_category_row_threshold=1.05)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_target_class_shift_threshold=-0.05)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_target_class_shift_threshold=1.01)

    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_outside_range_threshold=-0.5)

    # strictly positive numeric threshold (> 0)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_numeric_relative_shift_threshold=0.0)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_numeric_relative_shift_threshold=-1.5)

    # strictly positive integer thresholds (> 0)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_max_unseen_examples=0)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_max_unseen_examples=-5)
    with pytest.raises(InvalidConfigError):
        SanipyConfig(comparison_max_rows_for_overlap_check=0)

