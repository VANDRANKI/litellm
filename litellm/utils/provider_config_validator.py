"""Utilities for validating provider configuration before making API calls."""
from typing import Optional


def validate_api_key(api_key: Optional[str], provider: str) -> str:
    """Validate that an API key is present and non-empty.

    Args:
        api_key: The API key to validate, may be None.
        provider: The provider name for use in error messages.

    Returns:
        The validated, non-empty API key string.

    Raises:
        ValueError: If the API key is None or empty.
    """
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. "
            f"Set the environment variable or pass api_key explicitly."
        )
    return api_key


def validate_model_name(model: str, provider: str) -> str:
    """Validate that a model name is non-empty and properly formatted.

    Args:
        model: The model identifier string.
        provider: The provider name for use in error messages.

    Returns:
        The validated model string.

    Raises:
        ValueError: If the model string is empty or contains only whitespace.
    """
    model = model.strip()
    if not model:
        raise ValueError(
            f"Model name cannot be empty for provider '{provider}'."
        )
    return model


def validate_base_url(base_url: Optional[str], provider: str) -> Optional[str]:
    """Validate and normalize a custom base URL.

    Args:
        base_url: Optional custom API base URL.
        provider: The provider name for use in error messages.

    Returns:
        Normalized base URL (trailing slash removed) or None.

    Raises:
        ValueError: If the URL is provided but malformed (missing scheme).
    """
    if base_url is None:
        return None
    base_url = base_url.strip()
    if base_url and not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise ValueError(
            f"Invalid base_url for provider '{provider}': must start with http:// or https://. "
            f"Got: {base_url!r}"
        )
    return base_url.rstrip("/") if base_url else None


def validate_timeout(timeout: Optional[float], *, min_seconds: float = 1.0) -> Optional[float]:
    """Validate that a timeout value is positive and above a minimum threshold.

    Args:
        timeout: Timeout in seconds, or None to use the SDK default.
        min_seconds: Minimum allowed timeout. Defaults to 1.0.

    Returns:
        The validated timeout value, or None.

    Raises:
        ValueError: If timeout is not None and is less than `min_seconds`.
    """
    if timeout is None:
        return None
    if timeout < min_seconds:
        raise ValueError(
            f"Timeout must be at least {min_seconds} seconds, got {timeout}."
        )
    return timeout
