"""Tests for SanipyConfig."""

from sanipy.config import SanipyConfig


def test_default_config_creation():
    config = SanipyConfig()
    assert config.missing_low_threshold == 0.05
    assert config.high_cardinality_threshold == 50
    assert config.max_rows_for_expensive_checks == 100_000


def test_custom_config():
    config = SanipyConfig(
        missing_high_threshold=0.5,
        high_cardinality_threshold=100,
    )
    assert config.missing_high_threshold == 0.5
    assert config.high_cardinality_threshold == 100
    # Unmodified defaults
    assert config.missing_low_threshold == 0.05


def test_config_is_frozen():
    config = SanipyConfig()
    try:
        config.missing_low_threshold = 0.99  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
