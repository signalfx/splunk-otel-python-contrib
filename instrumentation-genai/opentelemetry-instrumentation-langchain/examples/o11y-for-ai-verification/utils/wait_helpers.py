import time
from typing import Callable, Any
from core.logger import get_logger


logger = get_logger(__name__)


class WaitHelpers:
    """Smart wait utilities for test synchronization."""

    @staticmethod
    def wait_for_condition(
        condition: Callable[[], bool],
        timeout: int = 30,
        interval: float = 1.0,
        error_message: str = "Condition not met within timeout",
    ) -> bool:
        """
        Wait for a condition to become true.

        Args:
            condition: Callable that returns bool
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            error_message: Error message if timeout

        Returns:
            True if condition met, raises TimeoutError otherwise

        Example:
            >>> wait_for_condition(lambda: len(get_traces()) > 0, timeout=60)
        """
        start_time = time.time()
        attempts = 0

        while time.time() - start_time < timeout:
            attempts += 1
            try:
                if condition():
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Condition met after {elapsed:.2f}s ({attempts} attempts)"
                    )
                    return True
            except Exception as e:
                logger.debug(f"Condition check failed: {e}")

            time.sleep(interval)

        elapsed = time.time() - start_time
        logger.error(
            f"Timeout after {elapsed:.2f}s ({attempts} attempts): {error_message}"
        )
        raise TimeoutError(error_message)

    @staticmethod
    def wait_for_trace(
        get_trace_func: Callable[[str], Any],
        trace_id: str,
        timeout: int = 120,
        interval: float = 5.0,
    ) -> Any:
        """
        Wait for trace to become available.

        Args:
            get_trace_func: Function to get trace
            trace_id: Trace ID to wait for
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            Trace data

        Raises:
            TimeoutError: If trace not available
        """

        def condition():
            try:
                trace = get_trace_func(trace_id)
                return trace is not None
            except Exception:
                return False

        WaitHelpers.wait_for_condition(
            condition,
            timeout=timeout,
            interval=interval,
            error_message=f"Trace {trace_id} not available after {timeout}s",
        )

        return get_trace_func(trace_id)

    @staticmethod
    def wait_for_element(
        page: Any, selector: str, timeout: int = 30, state: str = "visible"
    ) -> Any:
        """
        Wait for UI element (Playwright).

        Args:
            page: Playwright page object
            selector: Element selector
            timeout: Maximum wait time in milliseconds
            state: Element state ("visible", "attached", "hidden")

        Returns:
            Element locator
        """
        logger.info(f"Waiting for element: {selector} (state={state})")

        try:
            element = page.wait_for_selector(
                selector, timeout=timeout * 1000, state=state
            )
            logger.info(f"Element found: {selector}")
            return element
        except Exception as e:
            logger.error(f"Element not found: {selector} - {e}")
            raise

    @staticmethod
    def wait_for_page_load(page: Any, timeout: int = 30):
        """
        Wait for page to fully load (Playwright).

        Args:
            page: Playwright page object
            timeout: Maximum wait time in seconds
        """
        logger.info("Waiting for page load")

        try:
            page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            logger.info("Page loaded successfully")
        except Exception as e:
            logger.error(f"Page load timeout: {e}")
            raise

    @staticmethod
    def retry_on_exception(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 2.0,
        exceptions: tuple = (Exception,),
    ) -> Any:
        """
        Retry function on exception.

        Args:
            func: Function to retry
            max_attempts: Maximum retry attempts
            delay: Delay between retries in seconds
            exceptions: Exception types to catch

        Returns:
            Function result

        Raises:
            Last exception if all attempts fail
        """
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = func()
                if attempt > 1:
                    logger.info(f"Succeeded on attempt {attempt}")
                return result
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_attempts} attempts failed")

        raise last_exception
