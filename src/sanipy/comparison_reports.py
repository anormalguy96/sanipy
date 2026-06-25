"""DatasetComparisonReport — results from ``compare_train_test()``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    DiagnosticIssue,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)
from sanipy.reports import _json_sanitize


class DatasetComparisonReport:
    """Train/Test split health comparison report produced by :func:`compare_train_test`.

    Attributes:
        train_overview: Metadata about the training split.
        test_overview: Metadata about the testing split.
        issues: All detected comparison issues, sorted by severity.
        config: Configuration used for checks.
    """

    def __init__(
        self,
        issues: list[DiagnosticIssue],
        train_overview: dict[str, Any] | None = None,
        test_overview: dict[str, Any] | None = None,
        config: SanipyConfig | None = None,
    ) -> None:
        self.issues = sorted(
            issues, key=lambda i: i.severity_rank, reverse=True
        )
        self.train_overview = train_overview or {}
        self.test_overview = test_overview or {}
        self.config = config

    @property
    def score(self) -> int:
        """Heuristic health score from 0 (worst) to 100 (best)."""
        total_penalty = sum(issue.penalty for issue in self.issues)
        return max(0, 100 - total_penalty)

    def issues_by_severity(self, severity: str) -> list[DiagnosticIssue]:
        """Return issues matching the given severity level."""
        return [i for i in self.issues if i.severity == severity]

    def issues_by_category(self, category: str) -> list[DiagnosticIssue]:
        """Return issues matching the given category."""
        return [i for i in self.issues if i.category == category]

    @property
    def critical_issues(self) -> list[DiagnosticIssue]:
        """Shortcut for critical-severity issues."""
        return self.issues_by_severity(SEVERITY_CRITICAL)

    @property
    def high_issues(self) -> list[DiagnosticIssue]:
        """Shortcut for high-severity issues."""
        return self.issues_by_severity(SEVERITY_HIGH)

    def summary(self) -> str:
        """Return a human-readable text comparison report."""
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 56)
        lines.append("  Sanipy Train/Test Split Health Report")
        lines.append("=" * 56)
        lines.append("")

        # Train / Test info
        lines.append("  Splits Overview:")
        train_rows = self.train_overview.get("rows", "?")
        train_cols = self.train_overview.get("columns", "?")
        test_rows = self.test_overview.get("rows", "?")
        test_cols = self.test_overview.get("columns", "?")

        lines.append(f"    Train dataset: {train_rows:,} rows x {train_cols} columns"
                     if isinstance(train_rows, int)
                     else f"    Train dataset: {train_rows} rows x {train_cols} columns")
        lines.append(f"    Test dataset:  {test_rows:,} rows x {test_cols} columns"
                     if isinstance(test_rows, int)
                     else f"    Test dataset:  {test_rows} rows x {test_cols} columns")

        if isinstance(train_rows, int) and isinstance(test_rows, int) and train_rows > 0:
            ratio = test_rows / train_rows
            lines.append(f"    Row ratio:     {ratio:.2f} (Test/Train)")

        target = self.train_overview.get("target")
        if target:
            lines.append(f"    Target:        \"{target}\"")
        task = self.train_overview.get("task")
        if task:
            lines.append(f"    Task:          {task}")
        lines.append("")

        # Score
        lines.append(f"  Dataset score: {self.score}/100")
        lines.append("")

        # Disclaimer
        lines.append("  Note: This score is a heuristic estimate, not a")
        lines.append("  scientifically validated metric.")
        lines.append("")

        # Group by severity
        severity_groups = [
            (SEVERITY_CRITICAL, "[!!] Critical Issues"),
            (SEVERITY_HIGH, "[!]  High-Severity Issues"),
            (SEVERITY_MEDIUM, "[~]  Medium-Severity Issues"),
            (SEVERITY_LOW, "[.]  Low-Severity Issues"),
            (SEVERITY_INFO, "[i]  Info"),
        ]

        for severity, header in severity_groups:
            group = self.issues_by_severity(severity)
            if not group:
                continue
            lines.append(f"  {header}")
            lines.append(f"  {'-' * (len(header) + 2)}")
            for issue in group:
                lines.append(f"  - {issue.title}")
                if issue.recommendation:
                    lines.append(f"    -> {issue.recommendation}")
            lines.append("")

        # Count summary
        counts = {s: len(self.issues_by_severity(s)) for s, _ in severity_groups}
        non_zero = {s: c for s, c in counts.items() if c > 0}
        if non_zero:
            parts = [f"{c} {s}" for s, c in non_zero.items()]
            lines.append(f"  Total: {len(self.issues)} issues ({', '.join(parts)})")
        else:
            lines.append("  No issues detected. splits look consistent!")
        lines.append("")
        lines.append("=" * 56)
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the comparison report to a JSON-safe dictionary."""
        raw_dict = {
            "score": self.score,
            "train_overview": self.train_overview,
            "test_overview": self.test_overview,
            "total_issues": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        return _json_sanitize(raw_dict)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize to JSON. Optionally write to *path*."""
        data = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            try:
                Path(path).write_text(data, encoding="utf-8")
            except Exception as e:
                from sanipy.exceptions import ReportExportError
                raise ReportExportError(f"Failed to write JSON comparison report to {path}: {e}") from e
        return data

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Export the comparison report as a Markdown document."""
        lines: list[str] = []
        lines.append("# Sanipy Train/Test Split Health Report")
        lines.append("")
        lines.append(f"**Dataset score: {self.score}/100**")
        lines.append("")

        lines.append("## Splits Overview")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key, value in self.train_overview.items():
            if key in {"rows", "columns", "target", "task"}:
                label = f"Train {key.replace('_', ' ').title()}"
                val_escaped = str(value).replace("|", "\\|")
                lines.append(f"| {label} | {val_escaped} |")
        for key, value in self.test_overview.items():
            if key in {"rows", "columns"}:
                label = f"Test {key.replace('_', ' ').title()}"
                val_escaped = str(value).replace("|", "\\|")
                lines.append(f"| {label} | {val_escaped} |")
        lines.append("")

        lines.append("> **Note:** The dataset score is a heuristic estimate,")
        lines.append("> not a scientifically validated metric.")
        lines.append("")

        severity_groups = [
            (SEVERITY_CRITICAL, "Critical Issues"),
            (SEVERITY_HIGH, "High-Severity Issues"),
            (SEVERITY_MEDIUM, "Medium-Severity Issues"),
            (SEVERITY_LOW, "Low-Severity Issues"),
            (SEVERITY_INFO, "Informational"),
        ]

        for severity, header in severity_groups:
            group = self.issues_by_severity(severity)
            if not group:
                continue
            lines.append(f"## {header}")
            lines.append("")
            for issue in group:
                lines.append(f"- **{issue.title}**")
                if issue.recommendation:
                    lines.append(f"  - *Suggestion:* {issue.recommendation}")
                if issue.confidence != "high":
                    lines.append(f"  - *Confidence:* {issue.confidence}")
            lines.append("")

        lines.append("---")
        lines.append(
            f"*Report generated by Sanipy • "
            f"{len(self.issues)} issue(s) detected*"
        )
        lines.append("")

        md = "\n".join(lines)
        if path is not None:
            try:
                Path(path).write_text(md, encoding="utf-8")
            except Exception as e:
                from sanipy.exceptions import ReportExportError
                raise ReportExportError(f"Failed to write Markdown comparison report to {path}: {e}") from e
        return md

    def save(self, path: str | Path) -> None:
        """Save report to file. Format is inferred from extension."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".json":
            self.to_json(p)
        elif ext == ".md":
            self.to_markdown(p)
        elif ext == ".txt":
            try:
                p.write_text(self.summary(), encoding="utf-8")
            except Exception as e:
                from sanipy.exceptions import ReportExportError
                raise ReportExportError(f"Failed to write text comparison report to {path}: {e}") from e
        else:
            from sanipy.exceptions import ReportExportError
            raise ReportExportError(
                f"Unsupported file extension {ext!r}. "
                "Use .json, .md, or .txt."
            )

    def __repr__(self) -> str:
        return (
            f"DatasetComparisonReport(score={self.score}, issues={len(self.issues)})"
        )

    def __len__(self) -> int:
        return len(self.issues)

    def __bool__(self) -> bool:
        return len(self.issues) > 0
