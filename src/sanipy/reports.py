"""DatasetReport — aggregated results from ``check_dataset()``.

The report holds every detected issue, computes a heuristic health score,
and supports multiple export formats (text, dict, JSON, Markdown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sanipy.diagnostics import (
    DiagnosticIssue,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)


class DatasetReport:
    """Dataset health report produced by :func:`check_dataset`.

    Attributes:
        issues: All detected issues, sorted by severity (most severe first).
        dataset_info: Basic metadata about the analysed dataset.
        task: The ML task type (``"classification"``, ``"regression"``,
            or ``None``).
        target: Name of the target column, if provided.
    """

    def __init__(
        self,
        issues: list[DiagnosticIssue],
        dataset_info: dict[str, Any] | None = None,
        task: str | None = None,
        target: str | None = None,
    ) -> None:
        self.issues = sorted(
            issues, key=lambda i: i.severity_rank, reverse=True
        )
        self.dataset_info = dataset_info or {}
        self.task = task
        self.target = target

    # ── Score ────────────────────────────────────────────────────────

    @property
    def score(self) -> int:
        """Heuristic health score from 0 (worst) to 100 (best).

        This score is a simple convenience metric, **not** a
        scientifically validated measure. See the Sanipy documentation
        for how penalties are assigned.
        """
        total_penalty = sum(issue.penalty for issue in self.issues)
        return max(0, 100 - total_penalty)

    # ── Filtering helpers ───────────────────────────────────────────

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

    # ── Text summary ────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable text report."""
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 56)
        lines.append("  Sanipy Dataset Health Report")
        lines.append("=" * 56)
        lines.append("")

        # Dataset info
        if self.dataset_info:
            rows = self.dataset_info.get("rows", "?")
            cols = self.dataset_info.get("columns", "?")
            mem = self.dataset_info.get("memory_mb", "?")
            lines.append(f"  Dataset:  {rows:,} rows x {cols} columns"
                         if isinstance(rows, int)
                         else f"  Dataset:  {rows} rows x {cols} columns")
            lines.append(f"  Memory:   ~{mem} MB")
            if self.target:
                lines.append(f"  Target:   \"{self.target}\"")
            if self.task:
                lines.append(f"  Task:     {self.task}")
            lines.append("")

        # Score
        lines.append(f"  Dataset score: {self.score}/100")
        lines.append("")

        # Disclaimer
        lines.append(
            "  Note: This score is a heuristic estimate, not a"
        )
        lines.append(
            "  scientifically validated metric."
        )
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
            lines.append("  No issues detected. Dataset looks healthy!")
        lines.append("")
        lines.append("=" * 56)
        lines.append("")

        return "\n".join(lines)

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report to a dictionary."""
        return {
            "score": self.score,
            "task": self.task,
            "target": self.target,
            "dataset_info": self.dataset_info,
            "total_issues": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize to JSON. Optionally write to *path*."""
        data = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Export the report as a Markdown document."""
        lines: list[str] = []
        lines.append("# Sanipy Dataset Health Report")
        lines.append("")
        lines.append(f"**Dataset score: {self.score}/100**")
        lines.append("")

        if self.dataset_info:
            lines.append("## Dataset Overview")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for key, value in self.dataset_info.items():
                label = key.replace("_", " ").title()
                lines.append(f"| {label} | {value} |")
            if self.target:
                lines.append(f"| Target | `{self.target}` |")
            if self.task:
                lines.append(f"| Task | {self.task} |")
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
                    lines.append(
                        f"  - *Confidence:* {issue.confidence}"
                    )
            lines.append("")

        lines.append("---")
        lines.append(
            f"*Report generated by Sanipy v0.1.0 • "
            f"{len(self.issues)} issue(s) detected*"
        )
        lines.append("")

        md = "\n".join(lines)
        if path is not None:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def save(self, path: str | Path) -> None:
        """Save report to file. Format is inferred from extension.

        Supported extensions: ``.json``, ``.md``, ``.txt``.
        """
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".json":
            self.to_json(p)
        elif ext == ".md":
            self.to_markdown(p)
        elif ext == ".txt":
            p.write_text(self.summary(), encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file extension {ext!r}. "
                "Use .json, .md, or .txt."
            )

    # ── Dunder ──────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DatasetReport(score={self.score}, issues={len(self.issues)}, "
            f"task={self.task!r})"
        )

    def __len__(self) -> int:  # noqa: D105
        return len(self.issues)

    def __bool__(self) -> bool:  # noqa: D105
        """True if there are any issues."""
        return len(self.issues) > 0


# Backward compatibility alias
SanipyReport = DatasetReport
