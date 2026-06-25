"""Formatting utilities for Sanipy reports."""

from __future__ import annotations


def pct(value: float) -> str:
    """Format a float as a percentage string, e.g. ``0.456 → '45.6%'``."""
    return f"{value * 100:.1f}%"


def fmt_number(value: int | float) -> str:
    """Format a number with thousands separator."""
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"
