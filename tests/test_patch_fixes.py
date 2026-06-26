"""Regression tests for Sanipy 0.1.1 patch fixes.

Tests cover:
- python -m sanipy execution (subprocess)
- Report wording (no awkward -- separators)
- Tiny dataset heuristic behavior
- Public API import stability
- Backward compatibility aliases
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sanipy import (
    check_dataset,
    scan_dataset,
    compare_train_test,
    compare_datasets,
    SanipyConfig,
    DatasetReport,
    DatasetComparisonReport,
    DiagnosticIssue,
    Severity,
)
from sanipy._checks.identifier_columns import check_identifier_columns
from sanipy._checks.leakage_risk import check_possible_leakage_risk
from sanipy._checks.target_analysis import check_target_analysis
from sanipy._checks.constant_features import check_constant_features
from sanipy._checks.correlations import check_correlations


# ── python -m sanipy tests ─────────────────────────────────────────


class TestModuleExecution:
    """Verify python -m sanipy works via subprocess."""

    def test_python_m_sanipy_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "sanipy", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Sanipy" in result.stdout or "sanipy" in result.stdout
        assert "check" in result.stdout
        assert "compare" in result.stdout

    def test_python_m_sanipy_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "sanipy", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "0.1.1" in output

    def test_python_m_sanipy_check_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "sanipy", "check", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Run dataset sanity checks" in result.stdout

    def test_python_m_sanipy_compare_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "sanipy", "compare", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Compare train and test splits" in result.stdout


# ── Report wording tests ───────────────────────────────────────────


class TestReportWording:
    """Verify no awkward -- separators exist in diagnostic output."""

    def test_leakage_title_no_double_dash(self):
        """Leakage risk messages should not contain '--' separators."""
        rng = np.random.RandomState(42)
        target = rng.choice([0, 1], 200)
        df = pd.DataFrame({
            "feature": rng.normal(0, 1, 200),
            "final_result": target.astype(float),
            "target": target,
        })
        issues = check_possible_leakage_risk(df, SanipyConfig(), target="target")
        for issue in issues:
            assert " --" not in issue.title, (
                f"Issue title contains '--': {issue.title}"
            )

    def test_target_imbalance_title_no_double_dash(self):
        """Target imbalance messages should use ':' not '--'."""
        df = pd.DataFrame({
            "target": [0] * 95 + [1] * 5,
        })
        issues, _ = check_target_analysis(
            df, target="target", task="classification", config=SanipyConfig()
        )
        for issue in issues:
            if "imbalanced" in issue.title:
                assert " --" not in issue.title, (
                    f"Issue title contains '--': {issue.title}"
                )
                assert ": " in issue.title

    def test_single_class_title_no_double_dash(self):
        """Single-class message should use ':' not '--'."""
        df = pd.DataFrame({"target": [1, 1, 1, 1, 1]})
        issues, _ = check_target_analysis(
            df, target="target", task="classification", config=SanipyConfig()
        )
        single_class_issues = [i for i in issues if i.id == "target-single-class"]
        assert len(single_class_issues) == 1
        assert " --" not in single_class_issues[0].title
        assert ": " in single_class_issues[0].title

    def test_near_constant_title_no_double_dash(self):
        """Near-constant messages should use ':' not '--'."""
        df = pd.DataFrame({"col": [1] * 100 + [2]})
        issues = check_constant_features(df, SanipyConfig())
        for issue in issues:
            if "near-constant" in issue.title:
                assert " --" not in issue.title

    def test_correlation_recommendation_no_double_dash(self):
        """Correlation recommendation messages should use ':' not '--'."""
        target = np.arange(100, dtype=float)
        df = pd.DataFrame({
            "feature": target * 0.8 + np.random.RandomState(42).normal(0, 0.5, 100),
            "target": target,
        })
        issues = check_correlations(df, SanipyConfig(), target="target", task="regression")
        for issue in issues:
            assert " -- " not in issue.recommendation, (
                f"Recommendation contains '--': {issue.recommendation}"
            )

    def test_full_report_no_double_dash_in_titles(self):
        """Full check_dataset run should produce no '--' in issue titles."""
        rng = np.random.RandomState(42)
        target = rng.choice([0, 1], 200, p=[0.95, 0.05])
        df = pd.DataFrame({
            "id": range(200),
            "final_result": target.astype(float),
            "constant": [1] * 200,
            "feature": rng.normal(0, 1, 200),
            "target": target,
        })
        report = check_dataset(df, target="target", task="classification")
        for issue in report.issues:
            assert " --" not in issue.title, (
                f"Issue title still contains '--': {issue.title}"
            )


# ── Tiny dataset heuristic tests ──────────────────────────────────


class TestTinyDataset:
    """Verify small datasets don't produce overconfident warnings."""

    def test_tiny_dataset_integer_column_not_flagged_as_id(self):
        """A 5-row dataset with integers should NOT trigger an ID warning
        when the column name doesn't match ID patterns."""
        df = pd.DataFrame({
            "value": [10, 20, 30, 40, 50],
            "label": [0, 1, 0, 1, 0],
        })
        issues = check_identifier_columns(df, SanipyConfig())
        # Should not flag 'value' as ID (uniqueness-only, tiny dataset)
        id_issues = [i for i in issues if "id-column-value" == i.id]
        assert len(id_issues) == 0

    def test_tiny_dataset_named_id_column_still_flagged(self):
        """A column named 'id' should still be flagged even in tiny datasets.
        When name matches AND uniqueness is high, this is a high-confidence
        match and does NOT get the small-dataset qualifier."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        issues = check_identifier_columns(df, SanipyConfig())
        id_issues = [i for i in issues if "id-column-id" == i.id]
        assert len(id_issues) == 1
        # High-confidence match (name + uniqueness): no small-dataset qualifier
        assert id_issues[0].confidence == "high"

    def test_tiny_dataset_named_id_low_uniqueness_has_qualifier(self):
        """A column named 'uuid' with low uniqueness on a tiny dataset
        should be flagged with small-dataset qualifier (name-only match)."""
        df = pd.DataFrame({
            "uuid": [1, 1, 2, 2, 3],
            "value": [10, 20, 30, 40, 50],
        })
        issues = check_identifier_columns(df, SanipyConfig())
        id_issues = [i for i in issues if "id-column-uuid" == i.id]
        assert len(id_issues) == 1
        assert "small dataset" in id_issues[0].title.lower()
        assert id_issues[0].confidence == "medium"

    def test_normal_dataset_id_still_detected(self):
        """Normal-sized datasets should still detect ID columns by uniqueness."""
        df = pd.DataFrame({
            "record": range(100),
            "value": range(100),
        })
        issues = check_identifier_columns(df, SanipyConfig())
        # 'record' is integer with 100% uniqueness, should be flagged
        id_issues = [i for i in issues if "id-column-record" == i.id]
        assert len(id_issues) == 1
        # Should NOT have small-dataset qualifier
        assert "small dataset" not in id_issues[0].title.lower()

    def test_tiny_comparison_overlap_low_severity(self):
        """Tiny dataset comparison row overlap should be low severity."""
        train_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        test_df = pd.DataFrame({
            "a": [1, 2, 6],
            "b": [10, 20, 60],
        })
        report = compare_train_test(train_df, test_df)
        overlap_issues = [i for i in report.issues if i.id == "comparison-row-overlap"]
        if overlap_issues:
            assert overlap_issues[0].severity == "low"
            assert "small dataset" in overlap_issues[0].title.lower()

    def test_tiny_comparison_id_overlap_low_severity(self):
        """Tiny dataset ID overlap should be low severity."""
        train_df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        test_df = pd.DataFrame({
            "user_id": [1, 2, 6],
            "value": [10, 20, 60],
        })
        report = compare_train_test(train_df, test_df)
        id_overlap_issues = [
            i for i in report.issues
            if i.id.startswith("comparison-id-overlap")
        ]
        if id_overlap_issues:
            assert id_overlap_issues[0].severity == "low"
            assert "small dataset" in id_overlap_issues[0].title.lower()


# ── Public API import stability tests ─────────────────────────────


class TestPublicImports:
    """Verify all public imports and backward-compat aliases still work."""

    def test_public_imports(self):
        """All documented public imports must work."""
        from sanipy import (
            check_dataset,
            scan_dataset,
            compare_train_test,
            compare_datasets,
            SanipyConfig,
            DatasetReport,
            DatasetComparisonReport,
            DiagnosticIssue,
            Severity,
        )
        assert callable(check_dataset)
        assert callable(scan_dataset)
        assert callable(compare_train_test)
        assert callable(compare_datasets)
        assert SanipyConfig is not None
        assert DatasetReport is not None
        assert DatasetComparisonReport is not None
        assert DiagnosticIssue is not None
        assert Severity is not None

    def test_backward_compat_aliases(self):
        """Backward compatibility aliases must remain."""
        from sanipy import SanipyReport, Issue

        assert SanipyReport is DatasetReport
        assert Issue is DiagnosticIssue

    def test_version_is_0_1_1(self):
        import sanipy
        assert sanipy.__version__ == "0.1.1"
