# Changelog

All notable changes to **Sanipy** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-26

### Fixed
- Improved `__main__.py` module entrypoint for reliable `python -m sanipy` execution.
- Fixed report message formatting: replaced awkward `--` separators with `: ` for cleaner output across all terminals.
- Fixed README Markdown rendering issues and added clearer heuristic disclaimers.
- Reduced confusing ID-like column warnings for very small datasets (< 20 rows).
- Downgraded row overlap and ID overlap severity for tiny datasets in comparison checks.

### Tests
- Added regression tests for `python -m sanipy --help` and `--version` via subprocess.
- Added tests verifying no `--` separators remain in diagnostic output titles.
- Added tests for tiny dataset heuristic behavior (ID columns and comparison overlap).
- Added public import and backward-compatibility import regression tests.

## [0.1.0a1] - 2026-06-26

### Added
- Scaffolded project structure (`pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`).
- Core orchestrator function `check_dataset()` and alias `scan_dataset()`.
- Immutable configuration options with the frozen `SanipyConfig` class.
- Detailed reports with `DatasetReport` (including plain text, JSON, and Markdown exporters) and score calculations.
- Diagnostic models with `DiagnosticIssue` and `Severity` enum tiers.
- 11 ML-focused dataset checks implemented:
  - Dataset overview and dimensional diagnostics.
  - Missing value detection.
  - Fully duplicate row detection.
  - Constant/near-constant features analysis.
  - Identifier/ID-like column detection.
  - Categorical column high-cardinality analysis.
  - Classification/regression target distribution checks.
  - Outliers detection using Interquartile Range (IQR).
  - Feature skewness detection and log-transformation recommendations.
  - Feature-feature multicollinearity and feature-target correlation analysis.
  - Target leakage risk name and correlation heuristics.
- Unit and integration tests (92 passing tests).
- Realistic synthetic datasets and task examples in `examples/`.
- GitHub Actions CI workflow configuration.

### Added
- Custom exceptions hierarchy for public API errors (`SanipyError`, `InvalidDatasetError`, `InvalidTargetError`, `InvalidTaskError`, `InvalidConfigError`, `ReportExportError`).
- Config threshold validation checks in `SanipyConfig` for strict range, ordering, and type matching.
- Comprehensive edge-case test suite (`tests/test_edge_cases.py` containing 25+ distinct validation scenarios).

### Changed
- Refactored orchestrator to catch unexpected failures in individual checks safely and flag them as internal warning diagnostics, preventing total check execution crashes (with configurable debug `fail_fast` option).
- Hardened all internal checks against edge-cases including duplicate columns, non-string column names, MultiIndex structures, and infinite values (`np.inf`/`-np.inf`).
- Enabled clean export of NumPy data types, datetimes, NaNs, and Pandas `NA`/`NaT` types to dict, JSON, and Markdown formats.

### [0.1.0a2] - 2026-06-26

### Added
- Command Line Interface (CLI) supporting `sanipy check` subcommand.
- Terminal output formatting options (text, json, markdown).
- Integration pipeline CI/CD quality gate command-line flags (`--fail-on`, `--fail-fast`, `--debug`).
- Automated tests verifying CLI arguments, parsing, and execution.

### [0.1.0a3] - 2026-06-26

### Added
- Train/Test Split Diagnostics feature with `compare_train_test()` and `compare_datasets()` APIs.
- Dedicated `DatasetComparisonReport` model mirroring existing serialization interfaces.
- 11 split validation checks:
  - Basic split overview and row ratio analysis.
  - Schema mismatch (missing/extra features, missing targets).
  - Dtype mismatch across splits.
  - Missingness shift detection.
  - Unseen categories in test sets.
  - Missing categories from test sets.
  - Numeric range violations.
  - Numeric summary shifts (mean, median, std shifts).
  - Target distribution comparison (classification proportion shifts & regression summary shifts).
  - Exact row duplicates/leakage check with hashing-based performance limits.
  - Identifier/entity ID overlap checks.
- CLI subcommand `sanipy compare` with output formatters, export paths, fail gates, and debugging.
- Comprehensive API and CLI test suites under `tests/test_comparison.py`, `tests/test_comparison_edge_cases.py`, and `tests/test_cli_compare.py` (totaling 42 new test cases).
- Target auto-detection logic in dataset comparison.

## [0.1.0a4] - 2026-06-26

### Changed
- Corrected README installation guidance for PyPI/TestPyPI readiness.
- Updated roadmap to mark CLI and train/test comparison as completed.
- Cleaned stale release-candidate documentation references.

### Security
- Confirmed no registry tokens or credentials are stored in the repository.

## [0.1.0] - 2026-06-26

### Added
- First public PyPI release of Sanipy.
- Dataset sanity diagnostics with `check_dataset`.
- Train/test split diagnostics with `compare_train_test`.
- CLI commands: `sanipy check` and `sanipy compare`.
- JSON and Markdown report exports.
- Config validation, custom exceptions, and safe execution guards.

### Notes
- Sanipy is still early-stage software. APIs may change before `1.0`.




