"""Error handling utilities for the KWS package."""

from typing import Optional, Type, Any


class KWSError(Exception):
    """Base exception class for all KWS-related errors."""

    def __init__(self, message: str = "An error occurred in KWS"):
        self.message = message
        super().__init__(self.message)


class DatasetError(KWSError):
    """Raised when there are issues with the dataset."""

    def __init__(self, message: str = "Dataset error"):
        super().__init__(f"Dataset error: {message}")


class ModelError(KWSError):
    """Raised when there are issues with models."""

    def __init__(self, message: str = "Model error"):
        super().__init__(f"Model error: {message}")


class AudioProcessingError(KWSError):
    """Raised when there are issues with audio processing."""

    def __init__(self, message: str = "Audio processing error"):
        super().__init__(f"Audio processing error: {message}")


def handle_error(
    error: Exception,
    custom_error: Optional[Type[KWSError]] = None,
    msg: Optional[str] = None,
    re_raise: bool = True,
) -> Any:
    """Handle exceptions in a consistent way across the package.

    Args:
        error: The caught exception
        custom_error: Optional custom error type to raise
        msg: Optional custom message
        re_raise: Whether to re-raise the exception

    Raises:
        KWSError: If re_raise is True and a custom_error is provided
        The original exception: If re_raise is True and no custom_error is provided
    """
    from loguru import logger

    error_msg = msg if msg else str(error)
    logger.error(f"{error_msg}: {str(error)}")

    if re_raise:
        if custom_error:
            raise custom_error(error_msg) from error
        raise

    return None
