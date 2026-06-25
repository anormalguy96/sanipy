# Sanipy

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

For now (alpha version), install directly from GitHub:

```bash
pip install git+https://github.com/anormalguy96/sanipy.git@main
```

Once published on PyPI, you will be able to install it via:

```bash
pip install sanipy
```

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

---

## Heuristic Disclaimer

Sanipy uses heuristics, not proven theorems. It may produce:

- **False positives:** Flagging something that's actually fine.
- **False negatives:** Missing something that is a real problem.

All warnings use cautious language ("possible issue", "may", "consider"). The goal is to help you **notice** things, not to make decisions for you.

---

## Roadmap

### Beta (planned)

- [ ] CLI: `sanipy check data.csv --target churn`
- [ ] Train/test comparison: `compare_train_test(train_df, test_df)`
- [ ] Simple drift detection (KS test, PSI)
- [ ] HTML report export
- [ ] Polars DataFrame support

### Future

- [ ] scikit-learn pipeline integration
- [ ] GitHub Actions example
- [ ] Documentation website

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
