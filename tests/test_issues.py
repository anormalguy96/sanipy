"""Tests for Issue dataclass."""

from sanipy.issues import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    Issue,
)


def test_issue_creation():
    issue = Issue(
        id="test-001",
        title="Test issue",
        severity=SEVERITY_MEDIUM,
        category="test",
    )
    assert issue.id == "test-001"
    assert issue.severity == "medium"
    assert issue.columns == []
    assert issue.evidence == {}


def test_issue_penalty():
    assert Issue(id="a", title="", severity=SEVERITY_INFO, category="").penalty == 0
    assert Issue(id="b", title="", severity=SEVERITY_LOW, category="").penalty == 1
    assert Issue(id="c", title="", severity=SEVERITY_MEDIUM, category="").penalty == 4
    assert Issue(id="d", title="", severity=SEVERITY_HIGH, category="").penalty == 8
    assert Issue(id="e", title="", severity=SEVERITY_CRITICAL, category="").penalty == 15


def test_issue_severity_rank():
    issue_info = Issue(id="a", title="", severity=SEVERITY_INFO, category="")
    issue_crit = Issue(id="b", title="", severity=SEVERITY_CRITICAL, category="")
    assert issue_crit.severity_rank > issue_info.severity_rank


def test_issue_to_dict():
    issue = Issue(
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
