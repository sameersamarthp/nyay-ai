"""
Rate limiter for respectful web scraping.
"""

import random
import time
from contextlib import contextmanager
from threading import Lock
from typing import Generator

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Thread-safe rate limiter with jitter for web scraping."""

    def __init__(
        self,
        min_interval: float | None = None,
        max_interval: float | None = None,
    ):
        """Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests.
            max_interval: Maximum seconds (with jitter).
        """
        self.min_interval = min_interval or settings.MIN_REQUEST_INTERVAL
        self.max_interval = max_interval or settings.MAX_REQUEST_INTERVAL
        self._last_request_time: float = 0.0
        self._lock = Lock()

    def _get_wait_time(self) -> float:
        """Calculate wait time with jitter."""
        return random.uniform(self.min_interval, self.max_interval)

    def wait(self) -> None:
        """Wait for the appropriate time before the next request."""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_request_time

            wait_time = self._get_wait_time()
            remaining_wait = wait_time - elapsed

            if remaining_wait > 0:
                logger.debug(f"Rate limiting: waiting {remaining_wait:.2f}s")
                time.sleep(remaining_wait)

            self._last_request_time = time.time()

    @contextmanager
    def acquire(self) -> Generator[None, None, None]:
        """Context manager for rate-limited requests.

        Usage:
            with rate_limiter.acquire():
                response = requests.get(url)
        """
        self.wait()
        try:
            yield
        finally:
            pass

    def backoff(self, seconds: int | None = None) -> None:
        """Extended wait for rate limit responses (429).

        Args:
            seconds: Seconds to wait. Defaults to RATE_LIMIT_BACKOFF.
        """
        wait_time = seconds or settings.RATE_LIMIT_BACKOFF
        logger.warning(f"Rate limited! Backing off for {wait_time}s")
        time.sleep(wait_time)
        self._last_request_time = time.time()


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on server responses."""

    def __init__(
        self,
        min_interval: float | None = None,
        max_interval: float | None = None,
    ):
        """Initialize adaptive rate limiter."""
        super().__init__(min_interval, max_interval)
        self._consecutive_successes = 0
        self._consecutive_errors = 0
        self._current_interval = self.min_interval

    def record_success(self) -> None:
        """Record a successful request."""
        self._consecutive_successes += 1
        self._consecutive_errors = 0

        # Gradually decrease interval after consecutive successes
        if self._consecutive_successes >= 10:
            self._current_interval = max(
                self.min_interval,
                self._current_interval * 0.95,
            )

    def record_error(self, is_rate_limit: bool = False) -> None:
        """Record a failed request.

        Args:
            is_rate_limit: True if the error was a rate limit (429).
        """
        self._consecutive_errors += 1
        self._consecutive_successes = 0

        if is_rate_limit:
            # Significantly increase interval on rate limit
            self._current_interval = min(
                self.max_interval * 2,
                self._current_interval * 2,
            )
            logger.warning(f"Rate limit hit! Increasing interval to {self._current_interval:.2f}s")
        else:
            # Slight increase on other errors
            self._current_interval = min(
                self.max_interval,
                self._current_interval * 1.2,
            )

    def _get_wait_time(self) -> float:
        """Get current wait time with jitter."""
        jitter = random.uniform(0, self._current_interval * 0.2)
        return self._current_interval + jitter
