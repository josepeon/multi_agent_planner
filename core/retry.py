"""
Retry Logic with Exponential Backoff

Provides robust retry mechanisms for LLM API calls and other operations:
- Exponential backoff with jitter
- Configurable retry counts and delays
- Specific exception handling
- Decorator and context manager patterns

Usage:
    from core.retry import retry_with_backoff, RetryConfig
    
    @retry_with_backoff(max_retries=3)
    def call_llm():
        return client.chat("Hello")
    
    # Or use the context manager
    with RetryContext(max_retries=3) as ctx:
        result = ctx.execute(lambda: client.chat("Hello"))
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    jitter_factor: float = 0.1  # ±10% jitter

    # Exceptions that should trigger retry
    retryable_exceptions: list[type[Exception]] = field(default_factory=lambda: [
        ConnectionError,
        TimeoutError,
        OSError,
    ])

    # Specific error messages to retry on
    retryable_messages: list[str] = field(default_factory=lambda: [
        "rate limit",
        "too many requests",
        "service unavailable",
        "timeout",
        "connection reset",
        "internal server error",
        "502", "503", "504", "529",
    ])


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay for the next retry attempt with exponential backoff."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    exception: Exception,
    config: RetryConfig
) -> bool:
    """Determine if an exception should trigger a retry."""
    # Check exception type
    for exc_type in config.retryable_exceptions:
        if isinstance(exception, exc_type):
            return True

    # Check error message
    error_msg = str(exception).lower()
    for pattern in config.retryable_messages:
        if pattern.lower() in error_msg:
            return True

    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: list[type[Exception]] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """
    Decorator that adds retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        retryable_exceptions: List of exception types to retry on
        on_retry: Optional callback called on each retry with (exception, attempt)
    
    Example:
        @retry_with_backoff(max_retries=3)
        def call_api():
            return requests.get("https://api.example.com")
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    if retryable_exceptions:
        config.retryable_exceptions = retryable_exceptions

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(f"All {max_retries} retries exhausted for {func.__name__}")
                        raise

                    if not should_retry(e, config):
                        logger.warning(f"Non-retryable exception in {func.__name__}: {e}")
                        raise

                    delay = calculate_delay(attempt, config)
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s. Error: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry logic.
    
    Example:
        with RetryContext(max_retries=3) as ctx:
            result = ctx.execute(lambda: api_call())
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        self.attempts = 0
        self.last_exception: Exception | None = None

    def __enter__(self) -> 'RetryContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # Don't suppress exceptions

    def execute(self, func: Callable[[], T]) -> T:
        """Execute a function with retry logic."""
        for attempt in range(self.config.max_retries + 1):
            self.attempts = attempt + 1
            try:
                return func()
            except Exception as e:
                self.last_exception = e

                if attempt >= self.config.max_retries:
                    raise

                if not should_retry(e, self.config):
                    raise

                delay = calculate_delay(attempt, self.config)
                logger.info(f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.2f}s")
                time.sleep(delay)

        if self.last_exception:
            raise self.last_exception
        raise RuntimeError("Unexpected retry loop exit")


def retry_llm_call(
    func: Callable[[], T],
    max_retries: int = 3,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """
    Convenience function for retrying LLM API calls.
    
    Pre-configured with common LLM API error patterns.
    
    Example:
        result = retry_llm_call(lambda: client.chat("Hello"))
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=1.0,
        max_delay=30.0,
        retryable_messages=[
            "rate limit", "rate_limit", "ratelimit",
            "too many requests", "429",
            "service unavailable", "503",
            "timeout", "timed out",
            "connection", "network",
            "internal server error", "500",
            "bad gateway", "502",
            "gateway timeout", "504",
            "overloaded", "capacity",
        ]
    )

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt >= max_retries:
                raise

            if not should_retry(e, config):
                raise

            delay = calculate_delay(attempt, config)
            logger.info(f"LLM retry {attempt + 1}/{max_retries} after {delay:.2f}s. Error: {e}")

            if on_retry:
                on_retry(e, attempt + 1)

            time.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")
