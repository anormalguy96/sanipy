"""Custom exceptions for Sanipy."""


class SanipyError(Exception):
    """Base exception for Sanipy."""


class InvalidDatasetError(TypeError, SanipyError):
    """Raised when the input dataset is not a valid pandas DataFrame."""


class InvalidTargetError(TypeError, SanipyError):
    """Raised when the target argument has an invalid type."""


class InvalidTaskError(ValueError, SanipyError):
    """Raised when the task is not classification, regression, or None."""


class InvalidConfigError(ValueError, SanipyError):
    """Raised when SanipyConfig contains invalid threshold values."""


class ReportExportError(ValueError, SanipyError):
    """Raised when report export fails."""

