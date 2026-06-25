"""Sanipy — Basic usage example.

Messy dataset with missing values, duplicates, ID columns,
high-cardinality categoricals, and possible leakage risk.
"""

import numpy as np
import pandas as pd

from sanipy import check_dataset

# Create a synthetic dataset with common ML problems
rng = np.random.RandomState(42)
n = 1000

churn = rng.choice([0, 1], n, p=[0.88, 0.12])

# Introduce target leakage risk (high correlation + suspicious name)
churn_outcome = np.copy(churn)
leak_indices = rng.choice(n, int(n * 0.02), replace=False)
for idx in leak_indices:
    churn_outcome[idx] = 1 - churn_outcome[idx]

df = pd.DataFrame({
    "customer_id": range(n),
    "age": np.concatenate([
        rng.normal(35, 10, n - 5),
        np.array([150, -10, 200, 180, 160]),  # Outliers
    ]),
    "income": rng.lognormal(10, 1, n),
    "signup_source": rng.choice(["web", "mobile", "referral"], n),
    "region": [f"region_{i}" for i in rng.randint(0, 80, n)],
    "loyalty_score": np.concatenate([
        rng.normal(50, 15, n - 50),
        np.full(50, np.nan),  # Missing values
    ]),
    "is_active": [1] * (n - 3) + [0, 0, 0],  # Near-constant
    "churn_outcome": churn_outcome,           # Target leakage risk
    "churn": churn,
})

# Add some exact duplicate rows to make it messier
duplicates = df.sample(n=10, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# Run Sanipy checks
report = check_dataset(df, target="churn")

# Print the full report
print(report.summary())

# Access individual issues
print(f"\nTotal issues found: {len(report)}")
print(f"Critical issues: {len(report.critical_issues)}")
print(f"High issues: {len(report.high_issues)}")

# Export to different formats
report.to_markdown("sanipy_report.md")
report.to_json("sanipy_report.json")

print("\nDone! Reports saved to sanipy_report.md and sanipy_report.json")
