"""Private engine for running train/test comparison diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CONFIDENCE_HIGH,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_INFO,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy.comparison_reports import DatasetComparisonReport
from sanipy.exceptions import (
    InvalidConfigError,
    InvalidDatasetError,
    InvalidTargetError,
    InvalidTaskError,
)


def _safe_run(check_name: str, check_func, config: SanipyConfig, *args, **kwargs):
    """Safely run a check function, catching unexpected errors if fail_fast=False."""
    try:
        return check_func(*args, **kwargs)
    except Exception as e:
        if config.fail_fast:
            raise
        return [DiagnosticIssue(
            id=f"internal-error-comparison-{check_name}",
            title=f"Sanipy comparison check '{check_name}' failed safely. Please report this as a bug.",
            severity=SEVERITY_LOW,
            category="internal_error",
            evidence={
                "check_name": check_name,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
            recommendation="Please report this trace as an issue in the Sanipy repository.",
            confidence=CONFIDENCE_HIGH,
        )]


def _classify_dtype(series: pd.Series) -> str:
    """Classify a series dtype into simplified category string."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical/object/string"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "categorical/object/string"
    return "other"


def check_split_overview(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None,
    task: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Check split shapes, ratios, and flags empty splits."""
    issues = []
    train_rows = len(train_df)
    test_rows = len(test_df)
    train_cols = len(train_df.columns)
    test_cols = len(test_df.columns)

    if train_rows == 0:
        issues.append(DiagnosticIssue(
            id="comparison-overview-empty-train",
            title="Training dataset is empty.",
            severity=SEVERITY_CRITICAL,
            category="overview",
            recommendation="Provide a non-empty training dataset.",
        ))
    if test_rows == 0:
        issues.append(DiagnosticIssue(
            id="comparison-overview-empty-test",
            title="Test dataset is empty.",
            severity=SEVERITY_CRITICAL,
            category="overview",
            recommendation="Provide a non-empty test dataset.",
        ))
    if train_cols == 0:
        issues.append(DiagnosticIssue(
            id="comparison-overview-no-cols-train",
            title="Training dataset has no columns.",
            severity=SEVERITY_CRITICAL,
            category="overview",
            recommendation="Provide columns in the training dataset.",
        ))
    if test_cols == 0:
        issues.append(DiagnosticIssue(
            id="comparison-overview-no-cols-test",
            title="Test dataset has no columns.",
            severity=SEVERITY_CRITICAL,
            category="overview",
            recommendation="Provide columns in the test dataset.",
        ))

    if train_rows > 0 and test_rows > 0:
        ratio = test_rows / train_rows
        if test_rows < 10:
            issues.append(DiagnosticIssue(
                id="comparison-overview-small-test",
                title=f"Test dataset size is extremely small ({test_rows} rows).",
                severity=SEVERITY_HIGH,
                category="overview",
                evidence={"test_rows": test_rows, "train_rows": train_rows, "ratio": ratio},
                recommendation="Evaluation metrics will be highly unstable. Increase test set size.",
            ))
        elif ratio < 0.05:
            issues.append(DiagnosticIssue(
                id="comparison-overview-ratio-small",
                title=f"Test/Train row ratio is suspiciously small (ratio: {ratio:.4f}).",
                severity=SEVERITY_MEDIUM,
                category="overview",
                evidence={"test_rows": test_rows, "train_rows": train_rows, "ratio": ratio},
                recommendation="A very small test set may not be representative. Check split strategy.",
            ))
        elif ratio > 1.0:
            issues.append(DiagnosticIssue(
                id="comparison-overview-ratio-large",
                title=f"Test/Train row ratio is suspiciously large (ratio: {ratio:.4f}).",
                severity=SEVERITY_LOW,
                category="overview",
                evidence={"test_rows": test_rows, "train_rows": train_rows, "ratio": ratio},
                recommendation="Test dataset is larger than the training dataset. Ensure this is intentional.",
            ))

    return issues


def check_schema_mismatch(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare columns, checking for MultiIndex columns, duplicates, and missing columns."""
    issues = []

    train_mi = isinstance(train_df.columns, pd.MultiIndex)
    test_mi = isinstance(test_df.columns, pd.MultiIndex)

    if train_mi or test_mi:
        issues.append(DiagnosticIssue(
            id="comparison-schema-multiindex",
            title="Dataset has MultiIndex columns. This is not fully supported.",
            severity=SEVERITY_LOW,
            category="overview",
            recommendation="Consider flattening MultiIndex columns for better check compatibility.",
        ))

    if not train_mi:
        train_dups = train_df.columns[train_df.columns.duplicated()].unique().tolist()
        if train_dups:
            issues.append(DiagnosticIssue(
                id="comparison-schema-duplicate-train",
                title=f"Training dataset contains duplicate columns: {', '.join(map(str, train_dups))}.",
                severity=SEVERITY_MEDIUM,
                category="overview",
                evidence={"duplicate_columns": [str(c) for c in train_dups]},
                recommendation="Ensure all training columns have unique names.",
            ))

    if not test_mi:
        test_dups = test_df.columns[test_df.columns.duplicated()].unique().tolist()
        if test_dups:
            issues.append(DiagnosticIssue(
                id="comparison-schema-duplicate-test",
                title=f"Test dataset contains duplicate columns: {', '.join(map(str, test_dups))}.",
                severity=SEVERITY_MEDIUM,
                category="overview",
                evidence={"duplicate_columns": [str(c) for c in test_dups]},
                recommendation="Ensure all test columns have unique names.",
            ))

    train_cols_set = set(train_df.columns)
    test_cols_set = set(test_df.columns)

    train_only = train_cols_set - test_cols_set
    test_only = test_cols_set - train_cols_set

    # Target column is checked separately
    clean_train_only = train_only - ({target} if target is not None else set())
    clean_test_only = test_only - ({target} if target is not None else set())

    if clean_train_only:
        issues.append(DiagnosticIssue(
            id="comparison-schema-missing-test",
            title=f"Columns present in train but missing in test: {', '.join(map(str, sorted(list(clean_train_only))))}.",
            severity=SEVERITY_HIGH,
            category="schema",
            columns=sorted(list(clean_train_only)),
            evidence={"missing_columns": sorted(list(clean_train_only))},
            recommendation="Ensure the test set contains the same feature columns as the training set.",
        ))

    if clean_test_only:
        issues.append(DiagnosticIssue(
            id="comparison-schema-missing-train",
            title=f"Columns present in test but missing in train: {', '.join(map(str, sorted(list(clean_test_only))))}.",
            severity=SEVERITY_HIGH,
            category="schema",
            columns=sorted(list(clean_test_only)),
            evidence={"extra_columns": sorted(list(clean_test_only))},
            recommendation="Remove extra columns from the test set or add them to the training data.",
        ))

    if target is not None:
        if target not in train_df.columns:
            issues.append(DiagnosticIssue(
                id="comparison-schema-missing-target-train",
                title=f"Target column '{target}' is missing from the training dataset.",
                severity=SEVERITY_CRITICAL,
                category="schema",
                columns=[target],
                recommendation="Ensure the target column is present in the training split.",
            ))
        if target not in test_df.columns:
            issues.append(DiagnosticIssue(
                id="comparison-schema-missing-target-test",
                title=f"Target column '{target}' is missing from the test dataset.",
                severity=SEVERITY_CRITICAL,
                category="schema",
                columns=[target],
                recommendation="Ensure the target column is present in the test split.",
            ))

    return issues


def check_dtype_mismatch(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare simplified dtype categories across common columns."""
    issues = []
    for col in common_cols:
        if col == target:
            continue

        train_cat = _classify_dtype(train_df[col])
        test_cat = _classify_dtype(test_df[col])

        if train_cat != test_cat:
            issues.append(DiagnosticIssue(
                id=f"comparison-dtype-mismatch-{col}",
                title=f"Column '{col}' has mismatched dtype category between train ({train_cat}) and test ({test_cat}).",
                severity=SEVERITY_HIGH,
                category="schema",
                columns=[col],
                evidence={
                    "train_dtype": str(train_df[col].dtype),
                    "test_dtype": str(test_df[col].dtype),
                    "train_category": train_cat,
                    "test_category": test_cat,
                },
                recommendation="Ensure features have matching data types across both splits to avoid pipeline errors.",
            ))
    return issues


def check_missingness_shift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare missing rates between train and test splits."""
    issues = []
    for col in common_cols:
        if col == target:
            continue

        train_miss = float(train_df[col].isnull().mean())
        test_miss = float(test_df[col].isnull().mean())
        shift = abs(train_miss - test_miss)

        if shift > config.comparison_missingness_shift_threshold:
            issues.append(DiagnosticIssue(
                id=f"comparison-missingness-shift-{col}",
                title=(
                    f"Column '{col}' has much higher/different missingness in test than train. "
                    f"Train: {train_miss:.1%}, Test: {test_miss:.1%}"
                ),
                severity=SEVERITY_MEDIUM,
                category="missing_values",
                columns=[col],
                evidence={
                    "train_missing_fraction": train_miss,
                    "test_missing_fraction": test_miss,
                    "difference": shift,
                    "threshold": config.comparison_missingness_shift_threshold,
                },
                recommendation="Investigate if missingness is due to split selection bias or data extraction issues.",
            ))
    return issues


def check_categorical_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect unseen test categories and missing categories from test."""
    issues = []
    for col in common_cols:
        if col == target:
            continue

        dtype_cat = _classify_dtype(train_df[col])
        if dtype_cat not in {"categorical/object/string", "boolean"}:
            continue

        try:
            train_cats = set(train_df[col].dropna().unique())
            test_cats = set(test_df[col].dropna().unique())
        except Exception:
            try:
                train_cats = set(train_df[col].dropna().astype(str).unique())
                test_cats = set(test_df[col].dropna().astype(str).unique())
            except Exception:
                continue

        # Unseen categories in test
        unseen = test_cats - train_cats
        if unseen:
            try:
                is_unseen_mask = test_df[col].isin(unseen)
            except Exception:
                is_unseen_mask = test_df[col].astype(str).isin(unseen)

            unseen_row_count = int(is_unseen_mask.sum())
            unseen_row_frac = unseen_row_count / len(test_df) if len(test_df) > 0 else 0.0

            severity = SEVERITY_HIGH if unseen_row_frac >= config.comparison_unseen_category_row_threshold else SEVERITY_LOW
            unseen_examples = sorted(list(unseen), key=str)[:config.comparison_max_unseen_examples]
            unseen_examples = [str(x) for x in unseen_examples]

            issues.append(DiagnosticIssue(
                id=f"comparison-unseen-categories-{col}",
                title=(
                    f"Column '{col}' contains {len(unseen)} categories in test "
                    f"that were not present in train ({unseen_row_frac:.1%} of test rows affected)."
                ),
                severity=severity,
                category="cardinality",
                columns=[col],
                evidence={
                    "unseen_categories": unseen_examples,
                    "unseen_count": len(unseen),
                    "unseen_row_count": unseen_row_count,
                    "unseen_row_fraction": unseen_row_frac,
                    "train_unique_count": len(train_cats),
                    "test_unique_count": len(test_cats),
                },
                recommendation=(
                    "Model cannot generalize to categories unseen during training. "
                    "Group rare/unseen categories into an 'Other' bin, or review the split."
                ),
            ))

        # Missing categories from test (low severity info)
        missing_from_test = train_cats - test_cats
        if missing_from_test:
            missing_examples = sorted(list(missing_from_test), key=str)[:5]
            missing_examples = [str(x) for x in missing_examples]
            issues.append(DiagnosticIssue(
                id=f"comparison-missing-categories-{col}",
                title=f"Column '{col}' has {len(missing_from_test)} categories in train that are absent from test.",
                severity=SEVERITY_INFO,
                category="cardinality",
                columns=[col],
                evidence={
                    "missing_categories": missing_examples,
                    "missing_count": len(missing_from_test),
                },
                recommendation=(
                    "Informational only. This is common but means performance on "
                    "these categories will not be evaluated."
                ),
            ))

    return issues


def check_numeric_range_violations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect test values outside training bounds."""
    issues = []
    for col in common_cols:
        if col == target:
            continue

        if _classify_dtype(train_df[col]) != "numeric":
            continue

        train_series = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        test_series = test_df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if train_series.empty or test_series.empty:
            continue

        train_min = float(train_series.min())
        train_max = float(train_series.max())
        test_min = float(test_series.min())
        test_max = float(test_series.max())

        below = test_series < train_min
        above = test_series > train_max
        count_below = int(below.sum())
        count_above = int(above.sum())
        total_outside = count_below + count_above
        pct_outside = total_outside / len(test_series) if len(test_series) > 0 else 0.0

        if pct_outside > 0:
            severity = SEVERITY_MEDIUM if pct_outside >= config.comparison_outside_range_threshold else SEVERITY_LOW
            issues.append(DiagnosticIssue(
                id=f"comparison-numeric-outside-range-{col}",
                title=f"Column '{col}' has test values outside training range (train range: {train_min} to {train_max}).",
                severity=severity,
                category="outliers",
                columns=[col],
                evidence={
                    "train_min": train_min,
                    "train_max": train_max,
                    "test_min": test_min,
                    "test_max": test_max,
                    "count_below_train_min": count_below,
                    "count_above_train_max": count_above,
                    "percentage_outside_range": pct_outside,
                },
                recommendation="Possible train/test distribution mismatch. Models may extrapolate poorly outside training range. Check for outliers.",
            ))

    return issues


def check_numeric_summary_shift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare summary statistics to detect possible numeric distribution shifts."""
    issues = []
    for col in common_cols:
        if col == target:
            continue

        if _classify_dtype(train_df[col]) != "numeric":
            continue

        train_series = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        test_series = test_df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if train_series.empty or test_series.empty:
            continue

        train_mean = train_series.mean()
        test_mean = test_series.mean()
        train_median = train_series.median()
        test_median = test_series.median()
        train_std = train_series.std()
        test_std = test_series.std()

        def get_rel_shift(train_val, test_val):
            if pd.isna(train_val) or pd.isna(test_val):
                return 0.0
            if abs(train_val) <= 1e-9 and abs(test_val) <= 1e-9:
                return 0.0
            if abs(train_val) <= 1e-9:
                return float(abs(test_val))
            return float(abs(test_val - train_val) / abs(train_val))

        mean_shift = get_rel_shift(train_mean, test_mean)
        median_shift = get_rel_shift(train_median, test_median)
        std_shift = get_rel_shift(train_std, test_std)

        if mean_shift > config.comparison_numeric_relative_shift_threshold or median_shift > config.comparison_numeric_relative_shift_threshold:
            issues.append(DiagnosticIssue(
                id=f"comparison-numeric-summary-shift-{col}",
                title=f"Column '{col}' shows a possible numeric distribution shift.",
                severity=SEVERITY_MEDIUM,
                category="skewness",
                columns=[col],
                evidence={
                    "train_mean": float(train_mean) if pd.notna(train_mean) else None,
                    "test_mean": float(test_mean) if pd.notna(test_mean) else None,
                    "mean_relative_shift": mean_shift,
                    "train_median": float(train_median) if pd.notna(train_median) else None,
                    "test_median": float(test_median) if pd.notna(test_median) else None,
                    "median_relative_shift": median_shift,
                    "train_std": float(train_std) if pd.notna(train_std) else None,
                    "test_std": float(test_std) if pd.notna(test_std) else None,
                    "std_relative_shift": std_shift,
                },
                recommendation="Possible train/test distribution mismatch. Manual review recommended. Check if split was randomized.",
            ))

    return issues


def check_target_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task: str,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare target distribution (class proportions or regression metrics)."""
    issues = []
    if target not in train_df.columns or target not in test_df.columns:
        return issues

    train_col = train_df[target]
    test_col = test_df[target]

    train_miss = float(train_col.isnull().mean())
    test_miss = float(test_col.isnull().mean())

    if train_miss > 0.05 or test_miss > 0.05:
        issues.append(DiagnosticIssue(
            id="comparison-target-missingness-high",
            title=f"Target column '{target}' has high missingness rate (train: {train_miss:.1%}, test: {test_miss:.1%}).",
            severity=SEVERITY_HIGH,
            category="target",
            columns=[target],
            evidence={"train_missing_fraction": train_miss, "test_missing_fraction": test_miss},
            recommendation="Clean or impute/exclude rows with missing target values prior to model training.",
        ))

    train_clean = train_col.dropna()
    test_clean = test_col.dropna()

    if train_clean.empty or test_clean.empty:
        return issues

    if task == "classification":
        train_classes = set(train_clean.unique())
        test_classes = set(test_clean.unique())

        if len(train_classes) == 1:
            issues.append(DiagnosticIssue(
                id="comparison-target-single-class-train",
                title=f"Training target '{target}' has only 1 class. Model training will not be possible.",
                severity=SEVERITY_CRITICAL,
                category="target",
                columns=[target],
                recommendation="Ensure training target has at least 2 distinct classes.",
            ))

        if len(test_classes) == 1:
            issues.append(DiagnosticIssue(
                id="comparison-target-single-class-test",
                title=f"Test target '{target}' has only 1 class.",
                severity=SEVERITY_CRITICAL,
                category="target",
                columns=[target],
                recommendation="Ensure test target has at least 2 distinct classes.",
            ))

        unseen_classes = test_classes - train_classes
        if unseen_classes:
            issues.append(DiagnosticIssue(
                id="comparison-target-class-unseen-test",
                title=f"Test target '{target}' contains unseen classes not present in train: {list(unseen_classes)}.",
                severity=SEVERITY_CRITICAL,
                category="target",
                columns=[target],
                evidence={"unseen_classes": list(unseen_classes)},
                recommendation="Ensure target classes are aligned. Remove rows with unseen target classes from test split.",
            ))

        train_counts = train_clean.value_counts(normalize=True).to_dict()
        test_counts = test_clean.value_counts(normalize=True).to_dict()

        max_diff = 0.0
        worst_class = None
        for cls in train_classes:
            tr_prop = train_counts.get(cls, 0.0)
            te_prop = test_counts.get(cls, 0.0)
            diff = abs(tr_prop - te_prop)
            if diff > max_diff:
                max_diff = diff
                worst_class = cls

        if max_diff > config.comparison_target_class_shift_threshold:
            issues.append(DiagnosticIssue(
                id="comparison-target-class-shift",
                title=(
                    f"Target '{target}' class proportions differ significantly between splits. "
                    f"Max shift: {max_diff:.1%} for class '{worst_class}'."
                ),
                severity=SEVERITY_HIGH,
                category="target",
                columns=[target],
                evidence={
                    "max_shift": max_diff,
                    "worst_class": str(worst_class),
                    "train_proportions": {str(k): float(v) for k, v in train_counts.items()},
                    "test_proportions": {str(k): float(v) for k, v in test_counts.items()},
                    "threshold": config.comparison_target_class_shift_threshold,
                },
                recommendation="Ensure stratified splitting is used for classification tasks.",
            ))

    elif task == "regression":
        train_clean_num = train_clean.replace([np.inf, -np.inf], np.nan).dropna()
        test_clean_num = test_clean.replace([np.inf, -np.inf], np.nan).dropna()

        if train_clean_num.empty or test_clean_num.empty:
            return issues

        train_min, train_max = float(train_clean_num.min()), float(train_clean_num.max())
        test_min, test_max = float(test_clean_num.min()), float(test_clean_num.max())

        if test_min < train_min or test_max > train_max:
            issues.append(DiagnosticIssue(
                id="comparison-target-range-regression",
                title=f"Regression target '{target}' test values fall outside training range.",
                severity=SEVERITY_MEDIUM,
                category="target",
                columns=[target],
                evidence={
                    "train_min": train_min,
                    "train_max": train_max,
                    "test_min": test_min,
                    "test_max": test_max,
                },
                recommendation="Models cannot extrapolate reliably. Ensure training set covers target evaluation range.",
            ))

        train_mean = train_clean_num.mean()
        test_mean = test_clean_num.mean()
        train_median = train_clean_num.median()
        test_median = test_clean_num.median()
        train_std = train_clean_num.std()
        test_std = test_clean_num.std()
        train_skew = train_clean_num.skew()
        test_skew = test_clean_num.skew()

        def get_rel_shift(train_val, test_val):
            if pd.isna(train_val) or pd.isna(test_val):
                return 0.0
            if abs(train_val) <= 1e-9 and abs(test_val) <= 1e-9:
                return 0.0
            if abs(train_val) <= 1e-9:
                return float(abs(test_val))
            return float(abs(test_val - train_val) / abs(train_val))

        mean_shift = get_rel_shift(train_mean, test_mean)
        median_shift = get_rel_shift(train_median, test_median)

        if mean_shift > config.comparison_numeric_relative_shift_threshold or median_shift > config.comparison_numeric_relative_shift_threshold:
            issues.append(DiagnosticIssue(
                id="comparison-target-shift-regression",
                title=f"Regression target '{target}' shows possible distribution shift.",
                severity=SEVERITY_MEDIUM,
                category="target",
                columns=[target],
                evidence={
                    "train_mean": float(train_mean) if pd.notna(train_mean) else None,
                    "test_mean": float(test_mean) if pd.notna(test_mean) else None,
                    "mean_relative_shift": mean_shift,
                    "train_median": float(train_median) if pd.notna(train_median) else None,
                    "test_median": float(test_median) if pd.notna(test_median) else None,
                    "median_relative_shift": median_shift,
                    "train_std": float(train_std) if pd.notna(train_std) else None,
                    "test_std": float(test_std) if pd.notna(test_std) else None,
                    "train_skewness": float(train_skew) if pd.notna(train_skew) else None,
                    "test_skewness": float(test_skew) if pd.notna(test_skew) else None,
                },
                recommendation="Possible target distribution shift. Verify split selection bias.",
            ))

    return issues


def check_row_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Detect duplicate rows between train and test splits."""
    issues = []
    train_rows = len(train_df)
    test_rows = len(test_df)

    if train_rows == 0 or test_rows == 0:
        return issues

    if train_rows > config.comparison_max_rows_for_overlap_check or test_rows > config.comparison_max_rows_for_overlap_check:
        issues.append(DiagnosticIssue(
            id="comparison-row-overlap-skipped",
            title="Train/Test exact row overlap check was skipped due to large dataset size.",
            severity=SEVERITY_INFO,
            category="overview",
            evidence={
                "skipped": True,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "max_rows_threshold": config.comparison_max_rows_for_overlap_check,
            },
            recommendation="Increase comparison_max_rows_for_overlap_check in config for full overlap validation.",
        ))
        return issues

    try:
        train_hashes = set(pd.util.hash_pandas_object(train_df, index=False))
        test_hashes = pd.util.hash_pandas_object(test_df, index=False)
        overlap_mask = test_hashes.isin(train_hashes)
        overlapping_count = int(overlap_mask.sum())
    except Exception as e:
        return [DiagnosticIssue(
            id="comparison-row-overlap-failed",
            title="Exact row overlap check failed due to unhashable datatypes.",
            severity=SEVERITY_INFO,
            category="overview",
            evidence={"error": str(e)},
            recommendation="Ensure the dataset has standard datatypes.",
        )]

    if overlapping_count > 0:
        overlap_pct = overlapping_count / test_rows
        severity = SEVERITY_HIGH if overlap_pct > 0.01 else SEVERITY_MEDIUM
        issues.append(DiagnosticIssue(
            id="comparison-row-overlap",
            title=f"Exact row overlap detected: {overlapping_count} test rows appear in training set ({overlap_pct:.2%}).",
            severity=severity,
            category="duplicates",
            evidence={
                "overlapping_rows": overlapping_count,
                "overlap_fraction": overlap_pct,
                "skipped": False,
            },
            recommendation="Remove overlapping rows from the test set to avoid split leakage and optimistic evaluations.",
        ))

    return issues


def check_id_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    common_cols: list[str],
    target: str | None,
    config: SanipyConfig,
) -> list[DiagnosticIssue]:
    """Compare entity identifiers present in both splits to warn about leakages."""
    issues = []
    from sanipy._checks.identifier_columns import _name_looks_like_id

    for col in common_cols:
        if col == target:
            continue

        train_series = train_df[col]
        test_series = test_df[col]

        train_rows = len(train_df)
        test_rows = len(test_df)
        if train_rows == 0 or test_rows == 0:
            continue

        train_uniq = train_series.nunique(dropna=True) / train_rows
        test_uniq = test_series.nunique(dropna=True) / test_rows

        name_match = _name_looks_like_id(col, config)

        is_train_id = False
        if name_match and train_uniq >= config.id_uniqueness_threshold:
            is_train_id = True
        elif name_match:
            is_train_id = True
        elif train_uniq >= config.id_uniqueness_threshold:
            if train_series.dtype == "object" or pd.api.types.is_integer_dtype(train_series):
                is_train_id = True

        is_test_id = False
        if name_match and test_uniq >= config.id_uniqueness_threshold:
            is_test_id = True
        elif name_match:
            is_test_id = True
        elif test_uniq >= config.id_uniqueness_threshold:
            if test_series.dtype == "object" or pd.api.types.is_integer_dtype(test_series):
                is_test_id = True

        if is_train_id and is_test_id:
            try:
                train_ids = set(train_series.dropna().unique())
                test_ids = set(test_series.dropna().unique())
                overlap = train_ids.intersection(test_ids)
                n_overlap = len(overlap)
            except Exception:
                try:
                    train_ids = set(train_series.dropna().astype(str).unique())
                    test_ids = set(test_series.dropna().astype(str).unique())
                    overlap = train_ids.intersection(test_ids)
                    n_overlap = len(overlap)
                except Exception:
                    continue

            if n_overlap > 0:
                overlap_pct_test = n_overlap / len(test_ids) if len(test_ids) > 0 else 0.0
                issues.append(DiagnosticIssue(
                    id=f"comparison-id-overlap-{col}",
                    title=f"ID-like column '{col}' has {n_overlap} overlapping values between train and test.",
                    severity=SEVERITY_MEDIUM,
                    category="id_columns",
                    columns=[col],
                    evidence={
                        "overlapping_values": n_overlap,
                        "overlap_percentage_test": overlap_pct_test,
                    },
                    recommendation="Possible entity leakage risk. Manual review recommended.",
                ))

    return issues


def run_comparison_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    config: SanipyConfig | None = None,
) -> DatasetComparisonReport:
    """Run all train/test split sanity diagnostics."""
    if not isinstance(train_df, pd.DataFrame):
        raise InvalidDatasetError(
            f"Expected training dataset to be a pandas DataFrame, got {type(train_df).__name__}."
        )
    if not isinstance(test_df, pd.DataFrame):
        raise InvalidDatasetError(
            f"Expected testing dataset to be a pandas DataFrame, got {type(test_df).__name__}."
        )

    if config is not None and not isinstance(config, SanipyConfig):
        raise InvalidConfigError(
            f"config must be a SanipyConfig instance, got {type(config).__name__}."
        )

    if config is None:
        config = SanipyConfig()

    if target is not None:
        if not isinstance(target, (str, int, float, bool, tuple)):
            raise InvalidTargetError(
                f"target must be a string or column label, got type {type(target).__name__}."
            )

    all_issues: list[DiagnosticIssue] = []

    # 1. Overview Check
    overview_issues = _safe_run("overview", check_split_overview, config, train_df, test_df, target, task, config)
    all_issues.extend(overview_issues)

    # Resolve task auto-detection
    resolved_task = task
    is_task_autodetected = False

    if resolved_task is not None:
        if not isinstance(resolved_task, str):
            raise InvalidTaskError(
                f"task must be a string, got {type(resolved_task).__name__}."
            )
        normalized = resolved_task.strip().lower()
        if normalized not in {"classification", "regression"}:
            raise InvalidTaskError(
                f"task must be 'classification', 'regression', or None -- got {resolved_task!r}."
            )
        resolved_task = normalized
    else:
        # Auto-detect task if target is provided
        if target is not None:
            target_series = None
            if target in train_df.columns:
                target_series = train_df[target].dropna()
            elif target in test_df.columns:
                target_series = test_df[target].dropna()

            if target_series is not None and not target_series.empty:
                from sanipy._checks.target_analysis import _detect_task
                resolved_task = _detect_task(target_series, config)
                is_task_autodetected = True

    if is_task_autodetected and resolved_task is not None:
        all_issues.append(DiagnosticIssue(
            id="comparison-target-task-auto",
            title=f"Task auto-detected as '{resolved_task}' based on target column characteristics.",
            severity=SEVERITY_INFO,
            category="target",
            columns=[target] if target is not None else [],
            evidence={
                "detected_task": resolved_task,
                "auto_detect_max_classes": config.auto_detect_task_max_classes,
            },
            recommendation="If this is wrong, pass task='classification' or task='regression' explicitly.",
        ))

    # Early exit if either split is empty
    if train_df.empty or test_df.empty:
        train_overview = {
            "rows": len(train_df),
            "columns": len(train_df.columns),
            "target": target,
            "task": resolved_task,
        }
        test_overview = {
            "rows": len(test_df),
            "columns": len(test_df.columns),
        }
        return DatasetComparisonReport(
            issues=all_issues,
            train_overview=train_overview,
            test_overview=test_overview,
            config=config,
        )

    # 2. Schema Mismatch
    schema_issues = _safe_run("schema_mismatch", check_schema_mismatch, config, train_df, test_df, target, config)
    all_issues.extend(schema_issues)

    # Identify common columns (preserving order from train_df if possible)
    common_cols = [col for col in train_df.columns if col in test_df.columns]

    # Remove duplicates from common_cols list itself (if any column is duplicated in schema)
    seen = set()
    unique_common_cols = []
    for c in common_cols:
        if c not in seen:
            seen.add(c)
            unique_common_cols.append(c)

    # 3. Dtype Mismatch
    dtype_issues = _safe_run("dtype_mismatch", check_dtype_mismatch, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(dtype_issues)

    # 4. Missingness Shift
    missingness_issues = _safe_run("missingness_shift", check_missingness_shift, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(missingness_issues)

    # 5. Categorical Unseen/Missing Values
    cat_issues = _safe_run("categorical_values", check_categorical_values, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(cat_issues)

    # 6. Numeric Range Violations
    range_issues = _safe_run("numeric_range_violations", check_numeric_range_violations, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(range_issues)

    # 7. Numeric Summary Shift
    summary_shift_issues = _safe_run("numeric_summary_shift", check_numeric_summary_shift, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(summary_shift_issues)

    # 8. Target Distribution
    if target is not None and resolved_task is not None:
        target_issues = _safe_run("target_distribution", check_target_distribution, config, train_df, test_df, target, resolved_task, config)
        all_issues.extend(target_issues)

    # 9. Exact Row Overlap
    overlap_issues = _safe_run("row_overlap", check_row_overlap, config, train_df, test_df, config)
    all_issues.extend(overlap_issues)

    # 10. ID Overlap
    id_issues = _safe_run("id_overlap", check_id_overlap, config, train_df, test_df, unique_common_cols, target, config)
    all_issues.extend(id_issues)

    train_overview = {
        "rows": len(train_df),
        "columns": len(train_df.columns),
        "target": target,
        "task": resolved_task,
    }
    test_overview = {
        "rows": len(test_df),
        "columns": len(test_df.columns),
    }

    return DatasetComparisonReport(
        issues=all_issues,
        train_overview=train_overview,
        test_overview=test_overview,
        config=config,
    )
