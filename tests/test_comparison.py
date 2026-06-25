"""Tests for Sanipy Train/Test Split health comparison API."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sanipy import (
    compare_train_test,
    compare_datasets,
    SanipyConfig,
    DatasetComparisonReport,
    DiagnosticIssue,
    Severity,
)
from sanipy.exceptions import InvalidDatasetError, InvalidTaskError, InvalidTargetError


def test_compare_train_test_basic():
    """Verify basic comparison works on simple consistent DataFrames."""
    train_df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": ["a", "b", "c"]})
    test_df = pd.DataFrame({"feat1": [2, 3, 4], "feat2": ["b", "c", "d"]})

    report = compare_train_test(train_df, test_df)
    assert isinstance(report, DatasetComparisonReport)
    assert report.train_overview["rows"] == 3
    assert report.test_overview["rows"] == 3
    assert report.train_overview["columns"] == 2


def test_compare_datasets_alias():
    """Verify compare_datasets is a direct alias of compare_train_test."""
    train_df = pd.DataFrame({"feat1": [1, 2, 3]})
    test_df = pd.DataFrame({"feat1": [2, 3, 4]})

    report = compare_datasets(train_df, test_df)
    assert isinstance(report, DatasetComparisonReport)


def test_schema_mismatch():
    """Verify column mismatches and missing target are reported."""
    train_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4], "target": [0, 1]})
    test_df = pd.DataFrame({"feat1": [1, 2], "feat3": [5, 6]})  # feat2 is missing, feat3 is extra, target is missing

    report = compare_train_test(train_df, test_df, target="target")

    issues = {i.id for i in report.issues}
    assert "comparison-schema-missing-test" in issues  # feat2
    assert "comparison-schema-missing-train" in issues  # feat3
    assert "comparison-schema-missing-target-test" in issues

    # Target missing from train
    train_no_target = pd.DataFrame({"feat1": [1, 2]})
    test_ok = pd.DataFrame({"feat1": [1, 2], "target": [0, 1]})
    report_train_missing = compare_train_test(train_no_target, test_ok, target="target")
    assert any(i.id == "comparison-schema-missing-target-train" for i in report_train_missing.issues)


def test_dtype_mismatch():
    """Verify dtype category mismatches are reported."""
    train_df = pd.DataFrame({"col": [1, 2, 3]})
    test_df = pd.DataFrame({"col": ["a", "b", "c"]})

    report = compare_train_test(train_df, test_df)
    assert any(i.id == "comparison-dtype-mismatch-col" for i in report.issues)


def test_missingness_shift():
    """Verify significant shifts in missing value rates are flagged."""
    train_df = pd.DataFrame({"col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # 50% missing in test (5/10), 0% in train (0/10)
    test_df = pd.DataFrame({"col": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, np.nan]})

    config = SanipyConfig(comparison_missingness_shift_threshold=0.20)
    report = compare_train_test(train_df, test_df, config=config)

    assert any(i.id == "comparison-missingness-shift-col" for i in report.issues)


def test_unseen_categories():
    """Verify unseen category detection and missing train categories."""
    train_df = pd.DataFrame({"cat": ["A", "A", "B", "B", "C"]})
    test_df = pd.DataFrame({"cat": ["A", "B", "C", "D", "E"]})  # D and E are unseen, C is present, A and B present

    config = SanipyConfig(comparison_unseen_category_row_threshold=0.10)
    report = compare_train_test(train_df, test_df, config=config)

    unseen_issue = next(i for i in report.issues if i.id == "comparison-unseen-categories-cat")
    assert unseen_issue.severity == "high"  # 2/5 = 40% affected >= 10%
    assert unseen_issue.evidence["unseen_count"] == 2
    assert set(unseen_issue.evidence["unseen_categories"]) == {"D", "E"}


def test_missing_categories_info():
    """Verify categories in train missing from test are reported as info."""
    train_df = pd.DataFrame({"cat": ["A", "B", "C"]})
    test_df = pd.DataFrame({"cat": ["A", "B", "B"]})  # C is missing from test

    report = compare_train_test(train_df, test_df)
    missing_issue = next(i for i in report.issues if i.id == "comparison-missing-categories-cat")
    assert missing_issue.severity == "info"
    assert missing_issue.evidence["missing_count"] == 1
    assert missing_issue.evidence["missing_categories"] == ["C"]


def test_numeric_outside_range():
    """Verify test values outside train numeric min/max are flagged."""
    train_df = pd.DataFrame({"num": [10, 15, 20, 25, 30]})
    test_df = pd.DataFrame({"num": [5, 15, 20, 25, 35]})  # 5 is below, 35 is above (2/5 = 40% outside)

    config = SanipyConfig(comparison_outside_range_threshold=0.10)
    report = compare_train_test(train_df, test_df, config=config)

    issue = next(i for i in report.issues if i.id == "comparison-numeric-outside-range-num")
    assert issue.severity == "medium"
    assert issue.evidence["percentage_outside_range"] == 0.4
    assert issue.evidence["count_below_train_min"] == 1
    assert issue.evidence["count_above_train_max"] == 1


def test_numeric_summary_shift():
    """Verify relative shift of mean/median flags potential distribution mismatch."""
    train_df = pd.DataFrame({"num": [10.0, 10.0, 10.0, 10.0, 10.0]})
    test_df = pd.DataFrame({"num": [15.0, 15.0, 15.0, 15.0, 15.0]})  # Mean shift is 0.50 (50%)

    config = SanipyConfig(comparison_numeric_relative_shift_threshold=0.30)
    report = compare_train_test(train_df, test_df, config=config)

    assert any(i.id == "comparison-numeric-summary-shift-num" for i in report.issues)


def test_classification_target_shift():
    """Verify classification target proportion checks."""
    train_df = pd.DataFrame({"target": ["A", "A", "A", "A", "B", "B", "B", "B"]})  # 50/50 A/B
    test_df = pd.DataFrame({"target": ["A", "A", "A", "A", "A", "A", "A", "B"]})  # 87.5/12.5 A/B (shift = 37.5%)

    config = SanipyConfig(comparison_target_class_shift_threshold=0.20)
    report = compare_train_test(train_df, test_df, target="target", task="classification", config=config)

    assert any(i.id == "comparison-target-class-shift" for i in report.issues)

    # Test target with unseen class
    test_df_unseen = pd.DataFrame({"target": ["A", "B", "C"]})
    report_unseen = compare_train_test(train_df, test_df_unseen, target="target", task="classification")
    assert any(i.id == "comparison-target-class-unseen-test" for i in report_unseen.issues)


def test_regression_target_shift():
    """Verify regression target range and summary shifts."""
    train_df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test_df = pd.DataFrame({"target": [1.5, 2.5, 3.5, 4.5, 8.0]})  # Range violation (8.0 > 5.0) and shift

    config = SanipyConfig(comparison_numeric_relative_shift_threshold=0.10)
    report = compare_train_test(train_df, test_df, target="target", task="regression", config=config)

    issue_ids = {i.id for i in report.issues}
    assert "comparison-target-range-regression" in issue_ids
    assert "comparison-target-shift-regression" in issue_ids


def test_exact_row_overlap():
    """Verify duplicate row leakage is detected."""
    train_df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    test_df = pd.DataFrame({"feat1": [1, 7, 8], "feat2": [4, 9, 10]})  # [1, 4] is exact overlap (1/3 = 33.3%)

    report = compare_train_test(train_df, test_df)
    overlap_issue = next(i for i in report.issues if i.id == "comparison-row-overlap")
    assert overlap_issue.severity == "high"
    assert overlap_issue.evidence["overlapping_rows"] == 1


def test_id_overlap():
    """Verify identifier overlaps are flagged."""
    train_df = pd.DataFrame({"client_id": [101, 102, 103], "val": [1, 2, 3]})
    test_df = pd.DataFrame({"client_id": [103, 104, 105], "val": [4, 5, 6]})  # 103 is overlapping

    report = compare_train_test(train_df, test_df)
    assert any(i.id == "comparison-id-overlap-client_id" for i in report.issues)


def test_report_exports(tmp_path: Path):
    """Verify JSON, Markdown, and TXT save/export functions."""
    train_df = pd.DataFrame({"feat1": [1, 2], "feat2": ["A", "B"]})
    test_df = pd.DataFrame({"feat1": [2, 3], "feat2": ["B", "C"]})

    report = compare_train_test(train_df, test_df)

    # to_dict & to_json
    raw_dict = report.to_dict()
    assert isinstance(raw_dict, dict)
    assert "score" in raw_dict
    assert "train_overview" in raw_dict
    assert "issues" in raw_dict

    raw_json = report.to_json()
    parsed_json = json.loads(raw_json)
    assert parsed_json["score"] == report.score

    # to_markdown
    md = report.to_markdown()
    assert "# Sanipy Train/Test Split Health Report" in md
    assert "## Splits Overview" in md

    # save file writes
    json_file = tmp_path / "comp.json"
    md_file = tmp_path / "comp.md"
    txt_file = tmp_path / "comp.txt"

    report.save(json_file)
    report.save(md_file)
    report.save(txt_file)

    assert json_file.exists()
    assert md_file.exists()
    assert txt_file.exists()

    # Verify content
    assert "score" in json_file.read_text(encoding="utf-8")
    assert "# Sanipy" in md_file.read_text(encoding="utf-8")
    assert "Sanipy Train/Test" in txt_file.read_text(encoding="utf-8")
