"""Retry utilities for LLM API clients.

Provides exponential backoff retry logic for handling transient API errors
such as rate limits, connection errors, and server errors.
"""

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error is transient and should be retried.
    """
    error_type = type(error).__name__

    # Retryable error types by provider
    retryable_types = {
        # Anthropic
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "APIStatusError",  # 500, 502, 503, 504 errors
        # OpenAI
        "APITimeoutError",
        # Google
        "ResourceExhausted",  # Rate limit
        "ServiceUnavailable",
        "DeadlineExceeded",
        # Generic
        "ConnectionError",
        "TimeoutError",
        "Timeout",
    }

    if error_type in retryable_types:
        return True

    # Check error message for common retryable patterns
    error_msg = str(error).lower()
    retryable_patterns = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "capacity",
    ]

    return any(pattern in error_msg for pattern in retryable_patterns)


def get_retry_after(error: Exception) -> float | None:
    """Extract retry-after value from error if available.

    Args:
        error: The exception to check.

    Returns:
        Seconds to wait before retry, or None if not specified.
    """
    # Check for retry_after attribute (Anthropic SDK)
    if hasattr(error, "retry_after"):
        return error.retry_after

    # Check for response headers (various SDKs)
    if hasattr(error, "response") and error.response is not None:
        headers = getattr(error.response, "headers", {})
        if "retry-after" in headers:
            try:
                return float(headers["retry-after"])
            except (ValueError, TypeError):
                pass

    # Check error message for retry hints
    error_msg = str(error)
    if "retry after" in error_msg.lower():
        # Try to extract number from message
        import re
        match = re.search(r"retry after (\d+)", error_msg.lower())
        if match:
            return float(match.group(1))

    return None


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for adding retry logic to functions.

    Uses exponential backoff with jitter for retries.

    Args:
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries.
        on_retry: Optional callback called before each retry with (error, attempt, delay).

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(max_retries=3, retry_delay=2.0)
        def call_api(prompt: str) -> str:
            return api.call(prompt)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e):
                        # Not a retryable error, raise immediately
                        raise

                    if attempt >= max_retries:
                        # Exhausted all retries
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(retry_delay * (2 ** attempt), max_delay)

                    # Check if error specifies retry-after
                    retry_after = get_retry_after(e)
                    if retry_after is not None:
                        delay = max(delay, retry_after)

                    # Add small jitter to avoid thundering herd
                    import random
                    delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt, delay)

                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError(f"{func.__name__} failed with unknown error")

        return wrapper
    return decorator


def retry_api_call(
    func: Callable[P, R],
    *args: P.args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs: P.kwargs,
) -> R:
    """Execute a function with retry logic.

    This is a functional alternative to the decorator for one-off usage.

    Args:
        func: Function to call.
        *args: Positional arguments for func.
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs).

    Example:
        result = retry_api_call(api.call, prompt, max_retries=3)
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                raise

            if attempt >= max_retries:
                logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                raise

            delay = min(retry_delay * (2 ** attempt), max_delay)

            retry_after = get_retry_after(e)
            if retry_after is not None:
                delay = max(delay, retry_after)

            import random
            delay = delay * (0.5 + random.random())

            logger.warning(
                f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected retry failure")
