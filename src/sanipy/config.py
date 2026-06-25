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

    # ── Train/Test Comparison ───────────────────────────────────────
    comparison_missingness_shift_threshold: float = 0.20
    """Max absolute difference in missing rate between splits before warning."""

    comparison_unseen_category_row_threshold: float = 0.05
    """Row fraction threshold for unseen categories to determine severity."""

    comparison_numeric_relative_shift_threshold: float = 0.30
    """Relative difference in numeric summary stats to trigger a shift warning."""

    comparison_target_class_shift_threshold: float = 0.15
    """Max class proportion difference between splits before warning."""

    comparison_outside_range_threshold: float = 0.01
    """Fraction of test values outside training range to trigger a warning."""

    comparison_max_unseen_examples: int = 10
    """Maximum number of unseen category examples to include in evidence."""

    comparison_max_rows_for_overlap_check: int = 100_000
    """Skip exact row overlap checks if test split exceeds this size."""

    # ── Developer Settings ──────────────────────────────────────────
    fail_fast: bool = False
    """If True, raise unexpected internal check exceptions immediately."""

    def __post_init__(self) -> None:
        from sanipy.exceptions import InvalidConfigError

        # Percentage parameters (between 0.0 and 1.0 inclusive)
        pct_fields = {
            "missing_low_threshold": self.missing_low_threshold,
            "missing_medium_threshold": self.missing_medium_threshold,
            "missing_high_threshold": self.missing_high_threshold,
            "missing_critical_threshold": self.missing_critical_threshold,
            "duplicate_warn_threshold": self.duplicate_warn_threshold,
            "near_constant_threshold": self.near_constant_threshold,
            "id_uniqueness_threshold": self.id_uniqueness_threshold,
            "imbalance_majority_threshold": self.imbalance_majority_threshold,
            "imbalance_critical_threshold": self.imbalance_critical_threshold,
            "outlier_warn_threshold": self.outlier_warn_threshold,
            "high_correlation_threshold": self.high_correlation_threshold,
            "leakage_correlation_threshold": self.leakage_correlation_threshold,
            "comparison_missingness_shift_threshold": self.comparison_missingness_shift_threshold,
            "comparison_unseen_category_row_threshold": self.comparison_unseen_category_row_threshold,
            "comparison_target_class_shift_threshold": self.comparison_target_class_shift_threshold,
            "comparison_outside_range_threshold": self.comparison_outside_range_threshold,
        }

        for name, value in pct_fields.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidConfigError(
                    f"{name} must be numeric, got {type(value).__name__}."
                )
            if not (0.0 <= float(value) <= 1.0):
                raise InvalidConfigError(
                    f"{name} must be between 0.0 and 1.0 (inclusive), got {value}."
                )

        # Ordering check for missing values
        if not (self.missing_low_threshold < self.missing_medium_threshold <
                self.missing_high_threshold < self.missing_critical_threshold):
            raise InvalidConfigError(
                "Missing value thresholds must be strictly ordered: "
                "missing_low_threshold < missing_medium_threshold < "
                "missing_high_threshold < missing_critical_threshold."
            )

        # Positive numeric parameters (> 0)
        pos_num_fields = {
            "outlier_iqr_multiplier": self.outlier_iqr_multiplier,
            "skewness_warn_threshold": self.skewness_warn_threshold,
            "target_skewness_threshold": self.target_skewness_threshold,
            "comparison_numeric_relative_shift_threshold": self.comparison_numeric_relative_shift_threshold,
        }
        for name, value in pos_num_fields.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidConfigError(
                    f"{name} must be numeric, got {type(value).__name__}."
                )
            if float(value) <= 0:
                raise InvalidConfigError(
                    f"{name} must be strictly positive, got {value}."
                )

        # Positive integer parameters (> 0)
        pos_int_fields = {
            "high_cardinality_threshold": self.high_cardinality_threshold,
            "max_columns_for_correlation": self.max_columns_for_correlation,
            "max_rows_for_expensive_checks": self.max_rows_for_expensive_checks,
            "auto_detect_task_max_classes": self.auto_detect_task_max_classes,
            "comparison_max_unseen_examples": self.comparison_max_unseen_examples,
            "comparison_max_rows_for_overlap_check": self.comparison_max_rows_for_overlap_check,
        }
        for name, value in pos_int_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidConfigError(
                    f"{name} must be an integer, got {type(value).__name__}."
                )
            if value <= 0:
                raise InvalidConfigError(
                    f"{name} must be strictly positive, got {value}."
                )


        # Validate types for id_name_patterns
        if not isinstance(self.id_name_patterns, tuple):
            raise InvalidConfigError(
                "id_name_patterns must be a tuple of strings."
            )
        for pattern in self.id_name_patterns:
            if not isinstance(pattern, str):
                raise InvalidConfigError(
                    "id_name_patterns items must be strings."
                )

        # Validate types for fail_fast
        if not isinstance(self.fail_fast, bool):
            raise InvalidConfigError(
                f"fail_fast must be a boolean, got {type(self.fail_fast).__name__}."
            )
