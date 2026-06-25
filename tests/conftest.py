"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sanipy.config import SanipyConfig


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """A small, clean DataFrame with no issues."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "feature_a": rng.normal(50, 10, 100),
        "feature_b": rng.normal(0, 1, 100),
        "category": rng.choice(["cat", "dog", "bird"], 100),
        "target": rng.choice([0, 1], 100, p=[0.6, 0.4]),
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """An empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def default_config() -> SanipyConfig:
    """Default SanipyConfig."""
    return SanipyConfig()
