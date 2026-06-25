"""Command-line interface (CLI) for Sanipy."""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
import numpy as np

import sanipy
from sanipy import check_dataset, SanipyConfig
from sanipy.exceptions import SanipyError, ReportExportError
from sanipy.diagnostics import SEVERITY_ORDER


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code:
            0 = Success
            1 = Completed but failed quality gate, or unexpected internal error
            2 = Invalid CLI usage, file/parsing error, or custom SanipyError
    """
    parser = argparse.ArgumentParser(
        prog="sanipy",
        description="Sanipy: Lightweight sanity checks for ML datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sanipy {sanipy.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # sanipy check
    check_parser = subparsers.add_parser("check", help="Run dataset sanity checks", description="Run dataset sanity checks")

    check_parser.add_argument("file", help="Path to a local CSV file")
    check_parser.add_argument("--target", help="Optional target column name")
    check_parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Optional ML task type (classification or regression)",
    )
    check_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    check_parser.add_argument("--out", help="Optional path to write the report file")
    check_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise unexpected check exceptions immediately (useful for development)",
    )
    check_parser.add_argument(
        "--fail-on",
        choices=["low", "medium", "high", "critical"],
        help="Exit with code 1 if issues exist at or above this severity",
    )
    check_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full Python tracebacks for unexpected errors",
    )

    # Config options
    check_parser.add_argument(
        "--max-rows-for-expensive-checks",
        type=int,
        help="Sampling threshold row count",
    )
    check_parser.add_argument(
        "--max-columns-for-correlation",
        type=int,
        help="Correlation calculation column count guard",
    )
    check_parser.add_argument(
        "--missing-high-threshold",
        type=float,
        help="Missing high severity threshold (between 0.0 and 1.0)",
    )
    check_parser.add_argument(
        "--high-cardinality-threshold",
        type=int,
        help="Unique value count threshold for categoricals",
    )
    check_parser.add_argument(
        "--imbalance-majority-threshold",
        type=float,
        help="Majority class fraction threshold (between 0.0 and 1.0)",
    )

    # sanipy compare
    compare_parser = subparsers.add_parser("compare", help="Compare train and test splits", description="Compare train and test splits")
    compare_parser.add_argument("train_file", help="Path to local training CSV file")
    compare_parser.add_argument("test_file", help="Path to local testing CSV file")
    compare_parser.add_argument("--target", help="Optional target column name")
    compare_parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Optional ML task type (classification or regression)",
    )
    compare_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    compare_parser.add_argument("--out", help="Optional path to write the comparison report file")
    compare_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise unexpected check exceptions immediately (useful for development)",
    )
    compare_parser.add_argument(
        "--fail-on",
        choices=["low", "medium", "high", "critical"],
        help="Exit with code 1 if issues exist at or above this severity",
    )
    compare_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full Python tracebacks for unexpected errors",
    )

    try:
        args = parser.parse_args(args=argv)
    except SystemExit as e:
        return e.code

    if args.command is None:
        parser.print_help()
        return 2

    # Check command implementation
    if args.command == "check":
        # Positional path validations
        if os.path.isdir(args.file):
            sys.stderr.write(f"Error: Path '{args.file}' is a directory, not a file.\n")
            return 2

        if not os.path.exists(args.file):
            sys.stderr.write(f"Error: File '{args.file}' does not exist.\n")
            return 2

        if not args.file.lower().endswith(".csv"):
            sys.stderr.write(
                f"Error: File '{args.file}' has an unsupported extension. "
                "Only .csv files are supported.\n"
            )
            return 2

        # Check output path extension conflicts
        if args.out:
            ext = os.path.splitext(args.out)[1].lower()
            if args.format == "json" and ext in {".md", ".txt"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2
            if args.format == "markdown" and ext in {".json", ".txt"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2
            if args.format == "text" and ext in {".json", ".md"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2

        # Build SanipyConfig
        config_kwargs = {}
        if args.fail_fast:
            config_kwargs["fail_fast"] = True
        if args.max_rows_for_expensive_checks is not None:
            config_kwargs["max_rows_for_expensive_checks"] = args.max_rows_for_expensive_checks
        if args.max_columns_for_correlation is not None:
            config_kwargs["max_columns_for_correlation"] = args.max_columns_for_correlation
        if args.missing_high_threshold is not None:
            config_kwargs["missing_high_threshold"] = args.missing_high_threshold
        if args.high_cardinality_threshold is not None:
            config_kwargs["high_cardinality_threshold"] = args.high_cardinality_threshold
        if args.imbalance_majority_threshold is not None:
            config_kwargs["imbalance_majority_threshold"] = args.imbalance_majority_threshold

        try:
            config = SanipyConfig(**config_kwargs)
        except SanipyError as e:
            sys.stderr.write(f"Configuration Error: {e}\n")
            return 2

        # Load CSV and execute checks
        try:
            df = pd.read_csv(args.file)
        except pd.errors.EmptyDataError:
            sys.stderr.write(f"Error: File '{args.file}' is empty.\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"Error parsing CSV file '{args.file}': {e}\n")
            return 2

        try:
            report = check_dataset(
                df=df,
                target=args.target,
                task=args.task,
                config=config,
            )
        except SanipyError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 2
        except Exception as e:
            if args.debug:
                raise
            sys.stderr.write(f"Unexpected internal error: {e}\n")
            sys.stderr.write(
                "Please run with --debug to see the full traceback, or report this as a bug.\n"
            )
            return 1

        # Format output handling
        if args.out:
            try:
                report.save(args.out)
                print(f"Sanipy report written to {args.out}")
            except ReportExportError as e:
                sys.stderr.write(f"Export Error: {e}\n")
                return 2
        else:
            if args.format == "json":
                print(report.to_json())
            elif args.format == "markdown":
                print(report.to_markdown())
            else:
                print(report.summary())

        # Fail-on check
        if args.fail_on:
            fail_on_rank = SEVERITY_ORDER.get(args.fail_on.lower())
            if fail_on_rank is not None:
                for issue in report.issues:
                    issue_rank = SEVERITY_ORDER.get(issue.severity.lower(), 0)
                    if issue_rank >= fail_on_rank:
                        sys.stderr.write(
                            f"Fail-on Quality Gate: Found issue '{issue.id}' "
                            f"with severity '{issue.severity}' >= '{args.fail_on}'.\n"
                        )
                        return 1

    elif args.command == "compare":
        # Positional path validations
        for fpath, label in [(args.train_file, "Train"), (args.test_file, "Test")]:
            if os.path.isdir(fpath):
                sys.stderr.write(f"Error: {label} path '{fpath}' is a directory, not a file.\n")
                return 2
            if not os.path.exists(fpath):
                sys.stderr.write(f"Error: {label} file '{fpath}' does not exist.\n")
                return 2
            if not fpath.lower().endswith(".csv"):
                sys.stderr.write(
                    f"Error: {label} file '{fpath}' has an unsupported extension. "
                    "Only .csv files are supported.\n"
                )
                return 2

        # Check output path extension conflicts
        if args.out:
            ext = os.path.splitext(args.out)[1].lower()
            if args.format == "json" and ext in {".md", ".txt"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2
            if args.format == "markdown" and ext in {".json", ".txt"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2
            if args.format == "text" and ext in {".json", ".md"}:
                sys.stderr.write(
                    f"Error: Format '{args.format}' conflicts with output file extension '{ext}'.\n"
                )
                return 2

        # Build SanipyConfig
        config_kwargs = {}
        if args.fail_fast:
            config_kwargs["fail_fast"] = True

        try:
            config = SanipyConfig(**config_kwargs)
        except SanipyError as e:
            sys.stderr.write(f"Configuration Error: {e}\n")
            return 2

        # Load CSVs
        try:
            train_df = pd.read_csv(args.train_file)
        except pd.errors.EmptyDataError:
            sys.stderr.write(f"Error: Train file '{args.train_file}' is empty.\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"Error parsing CSV train file '{args.train_file}': {e}\n")
            return 2

        try:
            test_df = pd.read_csv(args.test_file)
        except pd.errors.EmptyDataError:
            sys.stderr.write(f"Error: Test file '{args.test_file}' is empty.\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"Error parsing CSV test file '{args.test_file}': {e}\n")
            return 2

        try:
            from sanipy.comparison import compare_train_test
            report = compare_train_test(
                train_df=train_df,
                test_df=test_df,
                target=args.target,
                task=args.task,
                config=config,
            )
        except SanipyError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 2
        except Exception as e:
            if args.debug:
                raise
            sys.stderr.write(f"Unexpected internal error: {e}\n")
            sys.stderr.write(
                "Please run with --debug to see the full traceback, or report this as a bug.\n"
            )
            return 1

        # Format output handling
        if args.out:
            try:
                report.save(args.out)
                print(f"Sanipy comparison report written to {args.out}")
            except ReportExportError as e:
                sys.stderr.write(f"Export Error: {e}\n")
                return 2
        else:
            if args.format == "json":
                print(report.to_json())
            elif args.format == "markdown":
                print(report.to_markdown())
            else:
                print(report.summary())

        # Fail-on check
        if args.fail_on:
            fail_on_rank = SEVERITY_ORDER.get(args.fail_on.lower())
            if fail_on_rank is not None:
                for issue in report.issues:
                    issue_rank = SEVERITY_ORDER.get(issue.severity.lower(), 0)
                    if issue_rank >= fail_on_rank:
                        sys.stderr.write(
                            f"Fail-on Quality Gate: Found issue '{issue.id}' "
                            f"with severity '{issue.severity}' >= '{args.fail_on}'.\n"
                        )
                        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
