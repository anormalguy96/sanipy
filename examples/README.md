# Sanipy Examples

This directory contains executable Python scripts and sample CSV datasets showing how to use **Sanipy**.

## Files Overview

### Python Scripts
- [basic_usage.py](file:///d:/github_repos/sanipy/examples/basic_usage.py) — Demonstrates basic end-to-end dataset validation using the Python API, accessing individual issues, and exporting report files.
- [classification_example.py](file:///d:/github_repos/sanipy/examples/classification_example.py) — Shows classification target analysis, checking for severe imbalance and potential leakage.
- [regression_example.py](file:///d:/github_repos/sanipy/examples/regression_example.py) — Demonstrates regression target validation, including outlier target analysis, correlation detection, and missingness metrics.
- [comparison_example.py](file:///d:/github_repos/sanipy/examples/comparison_example.py) — Shows split-consistency validation by comparing a training dataset against a testing dataset.

### Datasets
- [messy_classification.csv](file:///d:/github_repos/sanipy/examples/messy_classification.csv) — Sample tabular classification dataset with common ML errors.
- [train_classification.csv](file:///d:/github_repos/sanipy/examples/train_classification.csv) — Sample training split for comparison tests.
- [test_classification.csv](file:///d:/github_repos/sanipy/examples/test_classification.csv) — Sample testing split for comparison tests.

## How to Run

Before running the examples, ensure that **Sanipy** is installed in your Python environment:

```bash
pip install sanipy
```

Or install in editable mode from the repository root:

```bash
pip install -e .
```

To run any script, use the `python` or `py` command:

```bash
python examples/basic_usage.py
python examples/classification_example.py
python examples/regression_example.py
python examples/comparison_example.py
```
