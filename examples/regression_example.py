"""Sanipy — Regression task example."""

import numpy as np
import pandas as pd

from sanipy import check_dataset

# Create a regression dataset with common issues
rng = np.random.RandomState(42)
n = 800

# Simulate house prices
sqft = rng.normal(1500, 500, n).clip(300)
bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
lot_size = sqft * rng.uniform(1.5, 4.0, n)

# Target: highly skewed price distribution
price = np.exp(
    10 + 0.0005 * sqft + 0.1 * bedrooms + rng.normal(0, 0.3, n)
)

df = pd.DataFrame({
    "listing_id": range(n),
    "sqft": sqft,
    "bedrooms": bedrooms,
    "lot_size": lot_size,  # Highly correlated with sqft
    "sqft_copy": sqft + rng.normal(0, 0.1, n),  # Near-duplicate feature
    "year_built": rng.randint(1950, 2024, n),
    "garage": rng.choice([0, 1], n),
    "price": price,
})

# Add some missing values
df.loc[rng.choice(n, 40, replace=False), "year_built"] = np.nan

report = check_dataset(df, target="price", task="regression")
print(report.summary())

# Export as markdown
md = report.to_markdown()
print("\n--- Markdown preview (first 30 lines) ---\n")
for line in md.split("\n")[:30]:
    print(line)
