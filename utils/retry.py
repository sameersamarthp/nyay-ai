"""
Retry logic with exponential backoff using tenacity.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = settings.MAX_RETRIES
    min_wait: float = 1.0  # Minimum wait between retries (seconds)
    max_wait: float = 30.0  # Maximum wait between retries (seconds)
    exponential_base: float = 2.0  # Exponential backoff base


def with_retry(
    config: RetryConfig | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (
        requests.RequestException,
        requests.Timeout,
        requests.ConnectionError,
    ),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding retry logic with exponential backoff.

    Args:
        config: Retry configuration. Uses defaults if not provided.
        retryable_exceptions: Tuple of exceptions that trigger retry.

    Returns:
        Decorated function with retry logic.

    Usage:
        @with_retry()
        def fetch_page(url: str) -> requests.Response:
            return requests.get(url)

        @with_retry(RetryConfig(max_attempts=5))
        def another_function():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_exponential(
                multiplier=config.min_wait,
                max=config.max_wait,
                exp_base=config.exponential_base,
            ),
            retry=retry_if_exception_type(retryable_exceptions),
            before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_rate_limit(
    func: Callable[..., requests.Response],
) -> Callable[..., requests.Response]:
    """Decorator specifically for handling HTTP 429 rate limit responses.

    This decorator checks the response status and raises an exception
    if rate limited, allowing tenacity to retry.

    Args:
        func: Function that returns a requests.Response.

    Returns:
        Decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> requests.Response:
        response = func(*args, **kwargs)

        if response.status_code == 429:
            # Get retry-after header if available
            retry_after = response.headers.get("Retry-After")
            wait_time = int(retry_after) if retry_after else settings.RATE_LIMIT_BACKOFF
            raise RateLimitError(f"Rate limited. Retry after {wait_time}s", wait_time)

        return response

    return wrapper


class RateLimitError(Exception):
    """Exception raised when rate limited."""

    def __init__(self, message: str, retry_after: int):
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Suggested wait time in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after


class ScrapingError(Exception):
    """Base exception for scraping errors."""

    pass


class ParseError(ScrapingError):
    """Error parsing document content."""

    pass


class FetchError(ScrapingError):
    """Error fetching document from source."""

    pass
