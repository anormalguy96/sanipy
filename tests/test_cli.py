"""Tests for Sanipy CLI."""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from sanipy.cli import main
from sanipy.config import SanipyConfig
from sanipy.exceptions import InvalidConfigError


def test_cli_version(capsys):
    # 1. sanipy --version
    exit_code = main(["--version"])
    assert exit_code == 0 or exit_code == 2  # argparse version might sys.exit
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "0.1.0a3" in output


def test_cli_help(capsys):
    # 2. sanipy --help
    exit_code = main(["--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Available commands" in captured.out or "Available commands" in captured.err


def test_cli_check_help(capsys):
    # 3. sanipy check --help
    exit_code = main(["check", "--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Run dataset sanity checks" in captured.out or "Run dataset sanity checks" in captured.err


def test_cli_no_command(capsys):
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Available commands" in captured.out or "Available commands" in captured.err


def test_cli_valid_csv_no_target(tmp_path, capsys):
    # 4. Valid CSV with no target
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Sanipy Dataset Health Report" in captured.out


def test_cli_valid_csv_with_target(tmp_path, capsys):
    # 5. Valid CSV with target
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert 'Target:   "target"' in captured.out


def test_cli_classification_task(tmp_path, capsys):
    # 6. Valid CSV with classification task
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--task", "classification"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Task:     classification" in captured.out


def test_cli_regression_task(tmp_path, capsys):
    # 7. Valid CSV with regression task
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [10.5, 20.0, 30.2]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--task", "regression"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Task:     regression" in captured.out


def test_cli_json_stdout(tmp_path, capsys):
    # 8. JSON output to stdout
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--format", "json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"score":' in captured.out


def test_cli_markdown_stdout(tmp_path, capsys):
    # 9. Markdown output to stdout
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--format", "markdown"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Sanipy Dataset Health Report" in captured.out


def test_cli_json_to_file(tmp_path, capsys):
    # 10. JSON output to file
    csv_path = tmp_path / "data.csv"
    out_json = tmp_path / "report.json"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--format", "json", "--out", str(out_json)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Sanipy report written to {out_json}" in captured.out
    assert out_json.exists()
    assert '"score":' in out_json.read_text(encoding="utf-8")


def test_cli_markdown_to_file(tmp_path, capsys):
    # 11. Markdown output to file
    csv_path = tmp_path / "data.csv"
    out_md = tmp_path / "report.md"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--format", "markdown", "--out", str(out_md)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Sanipy report written to {out_md}" in captured.out
    assert out_md.exists()
    assert "# Sanipy Dataset Health Report" in out_md.read_text(encoding="utf-8")


def test_cli_file_not_found(capsys):
    # 12. File not found
    exit_code = main(["check", "non_existent_file.csv"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Error: File 'non_existent_file.csv' does not exist." in captured.err


def test_cli_directory_instead_of_file(tmp_path, capsys):
    # 13. Directory path instead of file
    exit_code = main(["check", str(tmp_path)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "is a directory, not a file." in captured.err


def test_cli_unsupported_extension(tmp_path, capsys):
    # 14. Unsupported extension
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("a,b,c\n1,2,3")
    exit_code = main(["check", str(txt_path)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "has an unsupported extension" in captured.err


def test_cli_invalid_task(tmp_path, capsys):
    # 15. Invalid task
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--task", "invalid_task"])
    assert exit_code == 2  # argparse choices validation
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_cli_invalid_output_path(tmp_path, capsys):
    # 16. Invalid output path or extension conflicts
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    df.to_csv(csv_path, index=False)

    # conflicting format and extension (json vs md)
    exit_code = main(["check", str(csv_path), "--format", "json", "--out", "report.md"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "conflicts with output file extension" in captured.err


def test_cli_fail_on_high_gate(tmp_path, capsys):
    # 17. --fail-on high returns 1 when high issue exists (e.g. imbalanced target)
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "a": range(100),
        "target": [0] * 95 + [1] * 5
    })
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--fail-on", "high"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Fail-on Quality Gate: Found issue" in captured.err


def test_cli_fail_on_critical_no_issues(tmp_path, capsys):
    # 18. --fail-on critical returns 0 when no critical issue exists (medium imbalance is ok)
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "a": range(100),
        "target": [0] * 90 + [1] * 10
    })
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--target", "target", "--fail-on", "critical"])
    assert exit_code == 0


def test_cli_fail_fast_config(tmp_path, capsys):
    # 19. --fail-fast passes config correctly
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--fail-fast"])
    assert exit_code == 0


def test_cli_does_not_mutate_csv(tmp_path):
    # 20. CLI does not mutate or damage input CSV
    csv_path = tmp_path / "data.csv"
    content = "a,b\n1,2\n3,4\n"
    csv_path.write_text(content)

    exit_code = main(["check", str(csv_path)])
    assert exit_code == 0
    assert csv_path.read_text() == content


def test_cli_invalid_config_error(tmp_path, capsys):
    # 21. CLI catches config SanipyError and returns code 2
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    exit_code = main(["check", str(csv_path), "--missing-high-threshold", "-0.5"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Configuration Error" in captured.err or "Error" in captured.err


def test_cli_unexpected_error_graceful(tmp_path, capsys, monkeypatch):
    # 22. CLI handles unexpected error gracefully unless --debug
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    # Mock check_dataset to throw unexpected RuntimeError
    def mock_check(*args, **kwargs):
        raise RuntimeError("Unexpected internal crash")

    import sanipy.cli
    monkeypatch.setattr(sanipy.cli, "check_dataset", mock_check)

    exit_code = main(["check", str(csv_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Unexpected internal error: Unexpected internal crash" in captured.err
    assert "Please run with --debug" in captured.err

    with pytest.raises(RuntimeError):
        main(["check", str(csv_path), "--debug"])
