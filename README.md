# Sanipy

[![CI](https://github.com/anormalguy96/sanipy/actions/workflows/ci.yml/badge.svg)](https://github.com/anormalguy96/sanipy/actions)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0a4-orange)](https://pypi.org/project/sanipy/)

**Lightweight sanity checks for ML datasets.**

Sanipy helps AI/ML engineers, data scientists, and Python developers quickly detect common dataset and machine learning pipeline problems **before training a model**.

It does **not** clean your data. It does **not** train models. It gives you a fast, human-readable health report so you can decide what to fix.

---

## Why Sanipy?

Most data validation tools are powerful but heavy. You need schemas, expectation suites, dashboards, or complex configuration before you get useful feedback.

Sanipy takes a different approach:

- **One function call** -- `check_dataset(df, target="churn")` -- and you're done.
- **Zero config required** (but fully configurable).
- **Human-readable report** -- not HTML dashboards, not JSON walls.
- **ML-focused heuristics** -- imbalance, leakage, ID columns, encoding warnings.
- **CPU-friendly** -- no GPU, no heavy models, no large dependencies.
- **Honest** -- says "possible issue" when uncertain, never overconfident claims.

---

## Installation

### From PyPI

```bash
pip install sanipy
```

### From GitHub

```bash
pip install git+https://github.com/anormalguy96/sanipy.git@main
```

Sanipy is currently in alpha. APIs may change before stable `1.0`.

**Dependencies:** Only `pandas` and `numpy`. That's it.

---

## Quick Start

```python
from sanipy import check_dataset

import pandas as pd

df = pd.read_csv("your_dataset.csv")

report = check_dataset(df, target="churn")
print(report.summary())
```

---

## Example Output

```text
========================================================
  Sanipy Dataset Health Report
========================================================

  Dataset:  1,000 rows x 8 columns
  Memory:   ~0.16 MB
  Target:   "churn"
  Task:     classification

  Dataset score: 71/100

  Note: This score is a heuristic estimate, not a
  scientifically validated metric.

  [!]  High-Severity Issues
  ---------------------------
  - Target "churn" is imbalanced -- majority class is 88.1%.
    -> Use stratified train/test split. Consider F1-score or
       ROC-AUC instead of accuracy.

  [~]  Medium-Severity Issues
  -----------------------------
  - Column "customer_id" looks like an ID column (uniqueness: 100.0%).
    -> Consider removing it before training.
  - Column "region" has high cardinality (80 unique values).
    -> Consider target encoding, frequency encoding, or grouping
       rare categories.
  - Column "income" contains 85 possible outliers (8.5%).
    -> Investigate outliers. Do not automatically remove.

  [.]  Low-Severity Issues
  --------------------------
  - Column "loyalty_score" has 5.0% missing values (50/1,000).
    -> Simple imputation or dropping rows may be acceptable.

  Total: 13 issues (1 high, 5 medium, 1 low, 6 info)

========================================================
```

---

## What It Checks

| # | Check | What it detects |
|---|---|---|
| 1 | **Dataset overview** | Empty data, tiny datasets, missing target column |
| 2 | **Missing values** | Per-column missing %, with severity tiers |
| 3 | **Duplicate rows** | Full row duplicates, percentage |
| 4 | **Constant columns** | Zero-variance and near-constant columns |
| 5 | **ID-like columns** | Columns that look like identifiers (name + uniqueness) |
| 6 | **High cardinality** | Categoricals with too many unique values for one-hot |
| 7 | **Target (classification)** | Class imbalance, metric suggestions, single-class |
| 8 | **Target (regression)** | Skewness, outliers, log-transform suggestions |
| 9 | **Outliers** | IQR-based detection with configurable thresholds |
| 10 | **Skewness** | Highly skewed features with transform suggestions |
| 11 | **Correlation** | Feature-feature multicollinearity, feature-target |
| 12 | **Leakage heuristics** | Suspicious names + high target correlation |

---

## What It Does NOT Do

Sanipy is **not** a replacement for:

- **Great Expectations / Pandera** -- schema-based validation suites
- **ydata-profiling / Sweetviz** -- comprehensive EDA reports
- **Evidently AI** -- production drift monitoring
- **SHAP / LIME** -- model explainability (requires a trained model)
- **scikit-learn** -- model training and evaluation

Sanipy does **not**:

- Automatically clean or transform your data
- Train models
- Generate dashboards
- Require a GPU
- Need complex setup

---

## Configuration

All thresholds are tunable:

```python
from sanipy import SanipyConfig, check_dataset

config = SanipyConfig(
    missing_high_threshold=0.3,
    high_cardinality_threshold=100,
    imbalance_majority_threshold=0.9,
    outlier_iqr_multiplier=2.0,
)

report = check_dataset(df, target="churn", config=config)
```

See [`config.py`](src/sanipy/config.py) for all available options.

---

## Export Formats

```python
# Text summary
print(report.summary())

# Python dictionary
data = report.to_dict()

# JSON (string or file)
report.to_json("report.json")

# Markdown
report.to_markdown("report.md")

# Auto-detect from extension
report.save("report.json")   # JSON
report.save("report.md")     # Markdown
report.save("report.txt")    # Plain text
```

---

## Comparing Train and Test Datasets

Sanipy allows you to compare training and test splits to detect shape inconsistencies, schema mismatches, dtype mismatches, missingness rate shifts, unseen categories, numeric range violations, numeric distribution shifts, target distribution shifts, row overlaps (leakage), and entity ID leaks.

This is a lightweight pre-model split sanity check, not a production drift monitoring tool.

### Python API

```python
from sanipy import compare_train_test

comparison = compare_train_test(
    train_df,
    test_df,
    target="churn",
    task="classification",
)
print(comparison.summary())
```

### CLI Example

```bash
sanipy compare train.csv test.csv --target churn --task classification
```

### What It Checks

- **Basic split overview**: Row/column shapes, row ratios, empty splits, extremely small test sizes.
- **Schema mismatch**: Column presence discrepancies, missing target in either split, duplicate names, MultiIndex columns.
- **Dtype mismatch**: Warns if columns have incompatible dtype categories (numeric, categorical/object/string, datetime, boolean, other).
- **Missingness shift**: Warns when the missing value rate difference for a column exceeds a threshold (default 20%).
- **Categorical unseen values**: Warns if the test set has categories never seen in training, with rows affected fraction.
- **Missing categories from test**: Info check when categories in training are completely absent from test.
- **Numeric range violations**: Warns if test values fall outside training min/max bounds.
- **Numeric summary shift**: Compares relative shifts in mean, median, and standard deviation (default 30% threshold).
- **Target distribution comparison**: Target missingness rates, class proportion differences (classification), or regression summary stat shifts.
- **Train/Test row overlap**: Identifies exact duplicate rows between splits (skipped for performance above 100K rows).
- **Identifier overlap**: Detects overlapping values in entity ID-like columns indicating split leakages.

### What It Does NOT Do

- It is **not** a production data drift monitoring tool (no statistical hypothesis testing like KS-test, PSI, Jensen-Shannon, Evidently-style monitoring).
- It does **not** generate dashboards or HTML reports.
- It does **not** perform automatic cleaning or model evaluation.
- It does **not** support GPU/deep learning dependencies or non-Pandas engines (like Spark/Dask/Polars).

---

## Command Line Usage

Sanipy provides a clean command-line interface (CLI) to run checks directly on CSV datasets from your terminal. Under the hood, it uses the exact same validation engine as the Python API.

### Basic Syntax
```bash
sanipy check <path_to_csv> [options]
```

### Examples
```bash
# 1. Run checks on data.csv and print a plain-text summary
sanipy check data.csv

# 2. Specify target column for supervised task auto-detection
sanipy check data.csv --target churn

# 3. Explicitly set target and ML task
sanipy check data.csv --target churn --task classification
sanipy check data.csv --target price --task regression

# 4. Print report as JSON or Markdown to stdout
sanipy check data.csv --target churn --format json
sanipy check data.csv --target churn --format markdown

# 5. Export report directly to a file (prints a short confirmation message)
sanipy check data.csv --target churn --format json --out report.json
sanipy check data.csv --target churn --format markdown --out report.md

# 6. CI/CD Gate: Exit with code 1 if any high or critical issues are found
sanipy check data.csv --target churn --fail-on high

# 7. Enable fail-fast checks (raises unexpected errors instantly for debugging)
sanipy check data.csv --target churn --fail-fast

# 8. Show tracebacks on unexpected errors
sanipy check data.csv --target churn --debug
```

### CLI Options
- `file` (Positional): Path to local CSV file. *Note: Only `.csv` files are supported.*
- `--target TARGET`: Target column name.
- `--task {classification,regression}`: Machine learning task.
- `--format {text,json,markdown}`: Output format (default: `text`).
- `--out PATH`: Output path to write the report.
- `--fail-fast`: Passes `fail_fast=True` into configuration.
- `--fail-on {low,medium,high,critical}`: Exit with code `1` if any issue severity matches or exceeds the specified level.
- `--debug`: Enable standard tracebacks on unexpected errors.
- `--version`: Print installed Sanipy version.
- `--help` / `check --help`: Show helpful command hints.

### Exit Codes
The CLI utilizes strict exit codes for integration pipelines:
- `0` = Command completed successfully.
- `1` = Completed successfully, but failed the quality gate (triggered by `--fail-on`) OR encountered an unexpected internal error.
- `2` = Invalid CLI usage (e.g. wrong arguments, missing subcommand), file error (missing/is a directory), CSV parsing failure, or standard Sanipy validation exceptions (`SanipyError`).

---

## Dataset Score

Sanipy computes a heuristic health score from 0 to 100:

| Severity | Penalty |
|---|---|
| Critical | -15 |
| High | -8 |
| Medium | -4 |
| Low | -1 |
| Info | 0 |

**Score = max(0, 100 - total penalties)**

> **Important:** This score is a convenience heuristic, not a scientifically validated metric. It helps you quickly gauge dataset health at a glance.

---

## Performance & Memory

Sanipy is designed for **16 GB RAM, CPU-only machines**:

- Minimal dependencies (pandas + numpy only)
- No GPU required
- Automatic sampling for datasets > 100K rows
- Column-count guards for expensive correlation checks
- No O(n^2) algorithms on rows
- Works well on: 1K, 10K, 100K rows x 100 columns

## Error Handling & Validation

Sanipy follows a strict, professional design for robustness and safety:

### 1. Invalid API Usage (Raises Exceptions)
Improper usage of the public API raises specific, clear exceptions:
- `InvalidDatasetError` (inherits from `TypeError`): Raised if the input `df` is not a `pandas.DataFrame`.
- `InvalidTargetError` (inherits from `TypeError`): Raised if `target` has an invalid type (e.g. not string-like).
- `InvalidTaskError` (inherits from `ValueError`): Raised if `task` is not `"classification"`, `"regression"`, or `None`.
- `InvalidConfigError` (inherits from `ValueError`): Raised if `SanipyConfig` is instantiated with invalid values (out-of-bounds percentages, negative integers, or unordered thresholds).
- `ReportExportError` (inherits from `ValueError`): Raised if report file export fails due to unsupported extensions or write failures.

### 2. Dataset Quality Problems (Reported as Issues)
Data problems do **not** crash the library. Instead, they are reported as `DiagnosticIssue` objects in the returned `DatasetReport`:
- Missing values, duplicate rows, constant columns
- Target column missing from the DataFrame
- All-null target column or single-class classification target
- Non-string column names (analyzed safely without crashing)

### 3. Safe Check Execution & Debug Mode
By default, if an individual check encounters an unexpected error, Sanipy does **not** fail the entire run. It catches the error, appends an informational issue with category `"internal_error"`, and continues executing other checks.

For debugging and developer diagnostics:
```python
# Fail-fast mode: raises unexpected check exceptions immediately
config = SanipyConfig(fail_fast=True)
report = check_dataset(df, target="churn", config=config)
```

### 4. Known Alpha Limitations
- **MultiIndex Columns**: MultiIndex layouts are not fully supported and will trigger an overview info diagnostic issue.
- **Large Datasets**: Datasets exceeding `max_rows_for_expensive_checks` (default 100K) are automatically sampled for performance.
- **Multi-collinearity Limits**: Very wide datasets with many columns will skip correlation calculations unless `max_columns_for_correlation` is adjusted.
- **Heuristic Errors**: Checks are heuristic and may produce false positives; verify warnings before taking action.

---

## Heuristic Disclaimer


Sanipy uses heuristics, not proven theorems. It may produce:

- **False positives:** Flagging something that's actually fine.
- **False negatives:** Missing something that is a real problem.

All warnings use cautious language ("possible issue", "may", "consider"). The goal is to help you **notice** things, not to make decisions for you.

---

## Roadmap

### Completed

- [x] Python API: `check_dataset`
- [x] Python API: `compare_train_test`
- [x] CLI: `sanipy check`
- [x] CLI: `sanipy compare`
- [x] JSON/Markdown export
- [x] Error handling and config validation

### Planned / Future

- [ ] Lightweight drift metrics (e.g. KS test, PSI)
- [ ] HTML report export
- [ ] Optional richer documentation website
- [ ] Possible Polars DataFrame support
- [ ] scikit-learn pipeline integration

---

## Development

```bash
# Clone
git clone https://github.com/anormalguy96/sanipy.git
cd sanipy

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Run example
python examples/basic_usage.py
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Write tests for new checks
4. Keep dependencies minimal
5. Follow the existing code style (type hints, docstrings)
6. Open a PR

---

## License

[MIT](LICENSE)
