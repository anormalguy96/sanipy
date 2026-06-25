"""Example demonstrating train/test health comparison diagnostics in Sanipy."""

from __future__ import annotations

import pandas as pd
from sanipy import compare_train_test


def main():
    # 1. Create a simulated training dataset
    train_df = pd.DataFrame({
        "client_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [20.0, 30.0, 40.0, 50.0, 25.0, 35.0, 45.0, 55.0, 28.0, 48.0],
        "income": [30000, 50000, 70000, 90000, 35000, 55000, 75000, 95000, 40000, 80000],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "churn": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    })

    # 2. Create a simulated test dataset with some anomalies:
    # - Overlapping client_id: 10
    # - Exact row overlap: row index 0 (client_id 1, age 20, etc.)
    # - Unseen category in 'region': 'Central'
    # - Out of range numeric in 'age': 99
    # - Different class proportions in 'churn' (50% churn instead of 30%)
    test_df = pd.DataFrame({
        "client_id": [1, 11, 12, 13, 10],
        "age": [20.0, 27.0, 99.0, 32.0, 48.0],
        "income": [30000, 42000, 85000, 49000, 80000],
        "region": ["North", "North", "Central", "East", "South"],
        "churn": [0, 0, 1, 1, 1]
    })

    print("Running Sanipy train/test comparison diagnostics...")
    report = compare_train_test(
        train_df=train_df,
        test_df=test_df,
        target="churn",
        task="classification"
    )

    # 3. Print report summary
    print(report.summary())

    # 4. Save to files
    report.save("comparison_report.md")
    report.save("comparison_report.json")
    print("Comparison reports saved to comparison_report.md and comparison_report.json")


if __name__ == "__main__":
    main()
