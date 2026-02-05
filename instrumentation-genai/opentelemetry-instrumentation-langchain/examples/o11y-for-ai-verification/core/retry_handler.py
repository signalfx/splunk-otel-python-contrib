import time
import functools
from typing import Callable, Tuple, Type, Any
from core.logger import get_logger


logger = get_logger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retry_on_status_codes: Tuple[int, ...] = (500, 502, 503, 504),
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        retry_on_exceptions: Exception types to retry on
        retry_on_status_codes: HTTP status codes to retry on

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_attempts=3, backoff_factor=2)
        ... def flaky_api_call():
        ...     response = requests.get('https://api.example.com/data')
        ...     return response.json()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except retry_on_exceptions as e:
                    if hasattr(e, "response") and hasattr(e.response, "status_code"):
                        status_code = e.response.status_code
                        if status_code not in retry_on_status_codes:
                            logger.warning(
                                f"Non-retryable status code {status_code}, not retrying"
                            )
                            raise

                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries ({max_attempts}) reached for {func.__name__}",
                            exc_info=True,
                        )
                        raise

                    wait_time = backoff_factor**attempt
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {wait_time}s...",
                        error=str(e),
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator
