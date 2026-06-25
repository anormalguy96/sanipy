"""ID-like column detection."""

from __future__ import annotations

import re

import pandas as pd

from sanipy.config import SanipyConfig
from sanipy.diagnostics import (
    CATEGORY_ID_COLUMNS,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    SEVERITY_MEDIUM,
    DiagnosticIssue,
)
from sanipy._utils.dataframe_ops import safe_get_series
from sanipy._utils.text_formatting import pct


# Patterns that strongly suggest an ID column (matched against the full name)
_EXACT_ID_NAMES: set[str] = {
    "id", "uuid", "guid", "index", "idx", "pk",
}

# Suffixes / substrings that suggest an ID column
_ID_SUFFIX_RE = re.compile(
    r"(?:_id|_uuid|_guid|_key|_index|_idx|_pk|_rowid|_row_num|_record_id)$",
    re.IGNORECASE,
)


def _name_looks_like_id(col_name: str | int | float | bool | tuple, config: SanipyConfig) -> bool:
    """Heuristic: does the column name suggest an ID?"""
    col_str = str(col_name)
    lower = col_str.strip().lower()

    # Exact match against known ID names
    if lower in _EXACT_ID_NAMES:
        return True

    # Check configured patterns
    for pattern in config.id_name_patterns:
        if lower == pattern.lower():
            return True

    # Suffix / substring match
    if _ID_SUFFIX_RE.search(lower):
        return True

    return False


def check_identifier_columns(
    df: pd.DataFrame,
    config: SanipyConfig,
    target: str | None = None,
) -> list[DiagnosticIssue]:
    """Detect columns that look like identifiers (IDs)."""
    issues: list[DiagnosticIssue] = []

    if df.empty:
        return issues

    n_rows = len(df)

    # Use unique column labels to avoid checking duplicate columns multiple times
    unique_cols = []
    seen = set()
    for col in df.columns:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)

    for col in unique_cols:
        # Never flag the target column as an ID
        if col == target:
            continue

        series = safe_get_series(df, col)
        uniqueness = series.nunique(dropna=True) / n_rows if n_rows > 0 else 0
        name_match = _name_looks_like_id(col, config)

        is_id = False
        evidence: dict = {
            "uniqueness_ratio": round(uniqueness, 4),
            "name_match": name_match,
        }

        if name_match and uniqueness >= config.id_uniqueness_threshold:
            is_id = True
            confidence = CONFIDENCE_HIGH
        elif name_match:
            is_id = True
            confidence = CONFIDENCE_MEDIUM
        elif uniqueness >= config.id_uniqueness_threshold:
            # High uniqueness but no name match — less confident
            # Only flag non-numeric or integer columns (strings with many
            # unique values are likely IDs; floats with high uniqueness
            # are just continuous features).
            if series.dtype == "object" or (
                pd.api.types.is_integer_dtype(series)
            ):
                is_id = True
                confidence = CONFIDENCE_MEDIUM
            else:
                continue
        else:
            continue

        if is_id:
            issues.append(DiagnosticIssue(
                id=f"id-column-{col}",
                title=(
                    f'Column "{col}" looks like an ID column '
                    f"(uniqueness: {pct(uniqueness)})."
                ),
                severity=SEVERITY_MEDIUM,
                category=CATEGORY_ID_COLUMNS,
                columns=[col],
                evidence=evidence,
                recommendation=(
                    f'Column "{col}" appears to be an identifier and is '
                    "unlikely to be a useful ML feature. Consider removing "
                    "it before training."
                ),
                confidence=confidence,
            ))

    return issues

