"""Tests for SanipyReport."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sanipy.issues import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    Issue,
)
from sanipy.report import SanipyReport


def _make_issues() -> list[Issue]:
    return [
        Issue(id="a", title="Critical problem", severity=SEVERITY_CRITICAL, category="test"),
        Issue(id="b", title="High problem", severity=SEVERITY_HIGH, category="test"),
        Issue(id="c", title="Medium problem", severity=SEVERITY_MEDIUM, category="test"),
        Issue(id="d", title="Low problem", severity=SEVERITY_LOW, category="test"),
        Issue(id="e", title="Info note", severity=SEVERITY_INFO, category="test"),
    ]


def test_score_calculation():
    issues = _make_issues()
    report = SanipyReport(issues=issues)
    # 100 - 15 - 8 - 4 - 1 - 0 = 72
    assert report.score == 72


def test_score_clamped_to_zero():
    # 10 critical issues → 150 penalty → clamped to 0
    issues = [
        Issue(id=f"c{i}", title="bad", severity=SEVERITY_CRITICAL, category="test")
        for i in range(10)
    ]
    report = SanipyReport(issues=issues)
    assert report.score == 0


def test_perfect_score():
    report = SanipyReport(issues=[])
    assert report.score == 100


def test_issues_sorted_by_severity():
    issues = _make_issues()
    report = SanipyReport(issues=issues)
    severities = [i.severity for i in report.issues]
    assert severities[0] == SEVERITY_CRITICAL
    assert severities[-1] == SEVERITY_INFO


def test_summary_output():
    issues = _make_issues()
    report = SanipyReport(
        issues=issues,
        dataset_info={"rows": 1000, "columns": 10, "memory_mb": 0.5},
        target="churn",
        task="classification",
    )
    text = report.summary()
    assert "Sanipy Dataset Health Report" in text
    assert "72/100" in text
    assert "Critical problem" in text
    assert "churn" in text


def test_to_dict():
    report = SanipyReport(issues=_make_issues(), task="classification")
    d = report.to_dict()
    assert d["score"] == 72
    assert d["total_issues"] == 5
    assert len(d["issues"]) == 5


def test_to_json():
    report = SanipyReport(issues=_make_issues())
    j = report.to_json()
    data = json.loads(j)
    assert data["score"] == 72


def test_to_json_file(tmp_path: Path):
    report = SanipyReport(issues=_make_issues())
    out = tmp_path / "report.json"
    report.to_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["total_issues"] == 5


def test_to_markdown():
    report = SanipyReport(
        issues=_make_issues(),
        dataset_info={"rows": 500, "columns": 5, "memory_mb": 0.1},
    )
    md = report.to_markdown()
    assert "# Sanipy Dataset Health Report" in md
    assert "72/100" in md
    assert "Critical Issues" in md


def test_to_markdown_file(tmp_path: Path):
    report = SanipyReport(issues=_make_issues())
    out = tmp_path / "report.md"
    report.to_markdown(out)
    assert out.exists()
    content = out.read_text()
    assert "Sanipy" in content


def test_save_json(tmp_path: Path):
    report = SanipyReport(issues=[])
    out = tmp_path / "report.json"
    report.save(out)
    assert out.exists()


def test_save_md(tmp_path: Path):
    report = SanipyReport(issues=[])
    out = tmp_path / "report.md"
    report.save(out)
    assert out.exists()


def test_save_txt(tmp_path: Path):
    report = SanipyReport(issues=[])
    out = tmp_path / "report.txt"
    report.save(out)
    assert out.exists()


def test_save_invalid_extension(tmp_path: Path):
    report = SanipyReport(issues=[])
    out = tmp_path / "report.csv"
    try:
        report.save(out)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_len_and_bool():
    assert len(SanipyReport(issues=[])) == 0
    assert not SanipyReport(issues=[])
    assert len(SanipyReport(issues=_make_issues())) == 5
    assert SanipyReport(issues=_make_issues())
