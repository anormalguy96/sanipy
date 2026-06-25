"""Sanipy — Classification task example."""

import numpy as np
import pandas as pd

from sanipy import SanipyConfig, check_dataset

# Create a classification dataset with severe imbalance
rng = np.random.RandomState(42)
n = 500

df = pd.DataFrame({
    "user_id": [f"USR-{i:04d}" for i in range(n)],
    "page_views": rng.poisson(10, n).astype(float),
    "session_duration": rng.exponential(300, n),
    "bounce_rate": rng.beta(2, 5, n),
    "is_mobile": rng.choice([0, 1], n),
    "final_status": rng.choice(["converted", "not_converted"], n, p=[0.05, 0.95]),
    "converted": rng.choice([0, 1], n, p=[0.96, 0.04]),  # Target: severe imbalance
})

# Use custom config with stricter thresholds
config = SanipyConfig(
    imbalance_majority_threshold=0.75,
    outlier_iqr_multiplier=2.0,
)

report = check_dataset(
    df,
    target="converted",
    task="classification",
    config=config,
)

print(report.summary())

# Show all issues as dicts
for issue in report.issues:
    if issue.severity in ("critical", "high"):
        print(f"  [{issue.severity.upper()}] {issue.title}")
        print(f"    → {issue.recommendation}")
        print()
