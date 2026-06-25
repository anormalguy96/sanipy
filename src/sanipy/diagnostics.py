"""DiagnosticIssue data model and Severity enum for Sanipy check results."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(str, enum.Enum):
    """Severity levels for dataset issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Severity levels (ordered low → high) ────────────────────────────
SEVERITY_INFO = "info"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

SEVERITY_ORDER: dict[str, int] = {
    SEVERITY_INFO: 0,
    SEVERITY_LOW: 1,
    SEVERITY_MEDIUM: 2,
    SEVERITY_HIGH: 3,
    SEVERITY_CRITICAL: 4,
}

# ── Severity → score penalty mapping ────────────────────────────────
SEVERITY_PENALTY: dict[str, int] = {
    SEVERITY_INFO: 0,
    SEVERITY_LOW: 1,
    SEVERITY_MEDIUM: 4,
    SEVERITY_HIGH: 8,
    SEVERITY_CRITICAL: 15,
}

# ── Confidence levels ───────────────────────────────────────────────
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"

# ── Issue categories ────────────────────────────────────────────────
CATEGORY_OVERVIEW = "overview"
CATEGORY_MISSING = "missing_values"
CATEGORY_DUPLICATES = "duplicates"
CATEGORY_CONSTANTS = "constant_columns"
CATEGORY_ID_COLUMNS = "id_columns"
CATEGORY_CARDINALITY = "cardinality"
CATEGORY_TARGET = "target"
CATEGORY_OUTLIERS = "outliers"
CATEGORY_SKEWNESS = "skewness"
CATEGORY_CORRELATION = "correlation"
CATEGORY_LEAKAGE = "leakage"
CATEGORY_PERFORMANCE = "performance"


@dataclass
class DiagnosticIssue:
    """A single issue detected by a Sanipy check.

    Attributes:
        id: Unique identifier, e.g. ``"missing-001"``.
        title: Human-readable one-line summary.
        severity: One of ``"info"``, ``"low"``, ``"medium"``,
            ``"high"``, ``"critical"``.
        category: Check category that produced this issue.
        columns: List of affected column names (may be empty).
        evidence: Machine-readable evidence dict.
        recommendation: Actionable suggestion for the user.
        confidence: How confident the heuristic is —
            ``"low"``, ``"medium"``, or ``"high"``.
    """

    id: str
    title: str
    severity: str
    category: str
    columns: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    confidence: str = CONFIDENCE_HIGH

    @property
    def penalty(self) -> int:
        """Score penalty associated with this issue's severity."""
        return SEVERITY_PENALTY.get(self.severity, 0)

    @property
    def severity_rank(self) -> int:
        """Numeric rank for sorting (higher = more severe)."""
        return SEVERITY_ORDER.get(self.severity, 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "columns": self.columns,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DiagnosticIssue(id={self.id!r}, severity={self.severity!r}, "
            f"title={self.title!r})"
        )


# Backward compatibility alias
Issue = DiagnosticIssue
