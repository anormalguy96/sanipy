# Changelog

All notable changes to **Sanipy** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
