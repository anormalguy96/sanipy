"""Sanipy configuration with tunable thresholds.

All thresholds have sensible defaults. Users can override any threshold
by passing a custom ``SanipyConfig`` to ``check_dataset()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SanipyConfig:
    """Configuration for Sanipy checks.

    All values can be overridden to tune sensitivity. The defaults are
    designed for typical tabular ML datasets (1K–100K rows, 10–100 cols).

    Example::

        config = SanipyConfig(
            missing_high_threshold=0.3,
            high_cardinality_threshold=100,
        )
        report = check_dataset(df, target="churn", config=config)
    """

    # ── Missing values ──────────────────────────────────────────────
    missing_low_threshold: float = 0.05
    """Fraction of missing values to trigger a *low* severity issue."""

    missing_medium_threshold: float = 0.15
    """Fraction of missing values to trigger a *medium* severity issue."""

    missing_high_threshold: float = 0.40
    """Fraction of missing values to trigger a *high* severity issue."""

    missing_critical_threshold: float = 0.70
    """Fraction of missing values to trigger a *critical* severity issue."""

    # ── Duplicates ──────────────────────────────────────────────────
    duplicate_warn_threshold: float = 0.01
    """Fraction of duplicate rows to raise a warning (default 1 %)."""

    # ── Constant / near-constant columns ────────────────────────────
    near_constant_threshold: float = 0.99
    """If a single value accounts for ≥ this fraction, flag as near-constant."""

    # ── ID-like columns ─────────────────────────────────────────────
    id_uniqueness_threshold: float = 0.95
    """Uniqueness ratio above which a column is suspected to be an ID."""

    id_name_patterns: tuple[str, ...] = field(default_factory=lambda: (
        "id", "uuid", "guid", "index", "row_id", "row_num",
        "record_id", "primary_key", "pk",
    ))
    """Name fragments (case-insensitive) that suggest an ID column."""

    # ── High-cardinality categoricals ───────────────────────────────
    high_cardinality_threshold: int = 50
    """Number of unique values above which a categorical is flagged."""

    # ── Target imbalance (classification) ───────────────────────────
    imbalance_majority_threshold: float = 0.80
    """Majority-class fraction above which the target is flagged as imbalanced."""

    imbalance_critical_threshold: float = 0.95
    """Majority-class fraction above which imbalance is *critical*."""

    # ── Outliers ────────────────────────────────────────────────────
    outlier_iqr_multiplier: float = 1.5
    """IQR multiplier for outlier fences (1.5 is the Tukey standard)."""

    outlier_warn_threshold: float = 0.05
    """Fraction of outlier values in a column to trigger a warning."""

    # ── Skewness ────────────────────────────────────────────────────
    skewness_warn_threshold: float = 2.0
    """Absolute skewness above which a column is flagged."""

    # ── Correlation ─────────────────────────────────────────────────
    high_correlation_threshold: float = 0.95
    """Absolute Pearson correlation between two features to flag."""

    leakage_correlation_threshold: float = 0.95
    """Absolute correlation with the target to flag as possible leakage."""

    max_columns_for_correlation: int = 100
    """Skip full correlation matrix if more numeric columns than this."""

    # ── Regression target ───────────────────────────────────────────
    target_skewness_threshold: float = 2.0
    """Absolute skewness of regression target to trigger a warning."""

    # ── Performance ─────────────────────────────────────────────────
    max_rows_for_expensive_checks: int = 100_000
    """Sample rows if the dataset exceeds this size for expensive checks."""

    # ── Task auto-detection ─────────────────────────────────────────
    auto_detect_task_max_classes: int = 20
    """If target has ≤ this many unique values, assume classification."""
