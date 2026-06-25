"""Tests for Sanipy CLI compare command."""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from sanipy.cli import main


def test_cli_compare_help(capsys):
    """Verify sanipy compare --help output."""
    exit_code = main(["compare", "--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Compare train and test splits" in output


def test_cli_compare_valid(tmp_path, capsys):
    """Verify basic comparison CLI run with text summary output."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Sanipy Train/Test Split Health Report" in captured.out


def test_cli_compare_target(tmp_path, capsys):
    """Verify comparison CLI with a target column."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4], "target": [1, 0, 1]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path), "--target", "target"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert 'Target:        "target"' in captured.out


def test_cli_compare_classification(tmp_path, capsys):
    """Verify classification task comparison via CLI."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4], "target": [1, 0, 1]}).to_csv(test_path, index=False)

    exit_code = main([
        "compare", str(train_path), str(test_path),
        "--target", "target", "--task", "classification"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Task:          classification" in captured.out


def test_cli_compare_regression(tmp_path, capsys):
    """Verify regression task comparison via CLI."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3], "target": [1.1, 2.2, 3.3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4], "target": [2.2, 3.3, 4.4]}).to_csv(test_path, index=False)

    exit_code = main([
        "compare", str(train_path), str(test_path),
        "--target", "target", "--task", "regression"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Task:          regression" in captured.out


def test_cli_compare_json_stdout(tmp_path, capsys):
    """Verify JSON output is printed to stdout."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path), "--format", "json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"score":' in captured.out


def test_cli_compare_markdown_stdout(tmp_path, capsys):
    """Verify Markdown output is printed to stdout."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path), "--format", "markdown"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# Sanipy Train/Test Split Health Report" in captured.out


def test_cli_compare_json_file(tmp_path, capsys):
    """Verify JSON file export works."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    out_json = tmp_path / "report.json"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4]}).to_csv(test_path, index=False)

    exit_code = main([
        "compare", str(train_path), str(test_path),
        "--format", "json", "--out", str(out_json)
    ])
    assert exit_code == 0
    assert out_json.exists()
    assert '"score":' in out_json.read_text(encoding="utf-8")


def test_cli_compare_markdown_file(tmp_path, capsys):
    """Verify Markdown file export works."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    out_md = tmp_path / "report.md"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2, 3, 4]}).to_csv(test_path, index=False)

    exit_code = main([
        "compare", str(train_path), str(test_path),
        "--format", "markdown", "--out", str(out_md)
    ])
    assert exit_code == 0
    assert out_md.exists()
    assert "# Sanipy Train/Test Split Health Report" in out_md.read_text(encoding="utf-8")


def test_cli_compare_file_not_found(tmp_path, capsys):
    """Verify error when a file is missing."""
    valid_csv = tmp_path / "valid.csv"
    pd.DataFrame({"a": [1]}).to_csv(valid_csv, index=False)

    exit_code = main(["compare", "non_existent.csv", str(valid_csv)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err

    exit_code2 = main(["compare", str(valid_csv), "non_existent2.csv"])
    assert exit_code2 == 2
    captured2 = capsys.readouterr()
    assert "does not exist" in captured2.err


def test_cli_compare_unsupported_extension(tmp_path, capsys):
    """Verify error on unsupported file extension."""
    valid_csv = tmp_path / "valid.csv"
    pd.DataFrame({"a": [1]}).to_csv(valid_csv, index=False)
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("1,2,3")

    exit_code = main(["compare", str(txt_path), str(valid_csv)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "has an unsupported extension" in captured.err


def test_cli_compare_directory_path(tmp_path, capsys):
    """Verify error when path is a directory."""
    valid_csv = tmp_path / "valid.csv"
    pd.DataFrame({"a": [1]}).to_csv(valid_csv, index=False)

    exit_code = main(["compare", str(tmp_path), str(valid_csv)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "is a directory" in captured.err


def test_cli_compare_invalid_task(tmp_path, capsys):
    """Verify choice error for task parameter."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path), "--task", "invalid_task"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_cli_compare_fail_on_high_gate(tmp_path, capsys):
    """Verify --fail-on flags quality gate exits."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    # Make target with unseen category in test -> triggers critical target issue
    pd.DataFrame({"target": ["A", "B", "A", "B"]}).to_csv(train_path, index=False)
    pd.DataFrame({"target": ["A", "B", "C", "D"]}).to_csv(test_path, index=False)

    exit_code = main([
        "compare", str(train_path), str(test_path),
        "--target", "target", "--task", "classification",
        "--fail-on", "critical"
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Fail-on Quality Gate: Found issue" in captured.err


def test_cli_compare_fail_fast(tmp_path):
    """Verify --fail-fast is propagated."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2]}).to_csv(test_path, index=False)

    exit_code = main(["compare", str(train_path), str(test_path), "--fail-fast"])
    assert exit_code == 0


def test_cli_compare_unexpected_error_graceful(tmp_path, capsys, monkeypatch):
    """Verify unexpected error handler and --debug behavior."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1]}).to_csv(train_path, index=False)
    pd.DataFrame({"a": [2]}).to_csv(test_path, index=False)

    def mock_compare(*args, **kwargs):
        raise RuntimeError("Unexpected comparison crash")

    import sanipy.cli
    # Use monkeypatch to patch the imported compare_train_test in sanipy.cli
    # Since main imports compare_train_test inside the main command block:
    # "from sanipy.comparison import compare_train_test"
    # We patch it in sanipy.comparison
    import sanipy.comparison
    monkeypatch.setattr(sanipy.comparison, "compare_train_test", mock_compare)

    exit_code = main(["compare", str(train_path), str(test_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Unexpected internal error: Unexpected comparison crash" in captured.err
    assert "Please run with --debug" in captured.err

    with pytest.raises(RuntimeError, match="Unexpected comparison crash"):
        main(["compare", str(train_path), str(test_path), "--debug"])
