"""Project-specific error types for user and internal failures."""


class KernupError(Exception):
    """Base class for internal Kernup exceptions."""


class UserError(KernupError):
    """Raised for actionable user-facing errors."""
