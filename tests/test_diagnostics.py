"""Tests for DiagnosticIssue dataclass and Severity enum."""

from sanipy.diagnostics import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
    Issue,
    Severity,
)


def test_severity_enum():
    assert Severity.INFO == "info"
    assert Severity.CRITICAL == "critical"
    assert isinstance(Severity.INFO, str)


def test_issue_creation():
    issue = DiagnosticIssue(
        id="test-001",
        title="Test issue",
        severity=SEVERITY_MEDIUM,
        category="test",
    )
    assert issue.id == "test-001"
    assert issue.severity == "medium"
    assert issue.columns == []
    assert issue.evidence == {}

    # Test compatibility alias
    issue_compat = Issue(
        id="test-001",
        title="Test issue",
        severity=SEVERITY_MEDIUM,
        category="test",
    )
    assert isinstance(issue_compat, DiagnosticIssue)


def test_issue_penalty():
    assert DiagnosticIssue(id="a", title="", severity=SEVERITY_INFO, category="").penalty == 0
    assert DiagnosticIssue(id="b", title="", severity=SEVERITY_LOW, category="").penalty == 1
    assert DiagnosticIssue(id="c", title="", severity=SEVERITY_MEDIUM, category="").penalty == 4
    assert DiagnosticIssue(id="d", title="", severity=SEVERITY_HIGH, category="").penalty == 8
    assert DiagnosticIssue(id="e", title="", severity=SEVERITY_CRITICAL, category="").penalty == 15


def test_issue_severity_rank():
    issue_info = DiagnosticIssue(id="a", title="", severity=SEVERITY_INFO, category="")
    issue_crit = DiagnosticIssue(id="b", title="", severity=SEVERITY_CRITICAL, category="")
    assert issue_crit.severity_rank > issue_info.severity_rank


def test_issue_to_dict():
    issue = DiagnosticIssue(
        id="test-001",
        title="Test",
        severity=SEVERITY_HIGH,
        category="test",
        columns=["col_a"],
        evidence={"value": 42},
        recommendation="Fix it.",
    )
    d = issue.to_dict()
    assert d["id"] == "test-001"
    assert d["columns"] == ["col_a"]
    assert d["evidence"]["value"] == 42
