"""
Base Page Object for O11y AI Test Framework.

Provides common functionality for all page objects including
navigation, element interaction, and wait helpers.
"""

from typing import Optional, List
from playwright.sync_api import Page, Locator, expect
from core.logger import get_logger


logger = get_logger(__name__)


class BasePage:
    """
    Base page object with common methods for all pages.
    """

    def __init__(self, page: Page, base_url: str):
        """
        Initialize base page.

        Args:
            page: Playwright page instance
            base_url: Application base URL
        """
        self.page = page
        self.base_url = base_url
        self.timeout = 30000  # 30 seconds default

    def navigate_to(self, path: str = "") -> None:
        """
        Navigate to specific path.

        Args:
            path: URL path relative to base_url
        """
        url = f"{self.base_url}{path}"
        logger.info(f"Navigating to: {url}")
        self.page.goto(url, wait_until="networkidle", timeout=self.timeout)

    def wait_for_load(self) -> None:
        """Wait for page to fully load."""
        self.page.wait_for_load_state("networkidle", timeout=self.timeout)
        logger.debug("Page loaded")

    def wait_for_selector(
        self, selector: str, timeout: Optional[int] = None
    ) -> Locator:
        """
        Wait for element to be visible.

        Args:
            selector: CSS selector
            timeout: Optional timeout in milliseconds

        Returns:
            Locator for the element
        """
        timeout = timeout or self.timeout
        locator = self.page.locator(selector)
        locator.wait_for(state="visible", timeout=timeout)
        logger.debug(f"Element visible: {selector}")
        return locator

    def click(self, selector: str, timeout: Optional[int] = None) -> None:
        """
        Click element.

        Args:
            selector: CSS selector
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        logger.debug(f"Clicking: {selector}")
        self.page.click(selector, timeout=timeout)

    def fill(self, selector: str, value: str, timeout: Optional[int] = None) -> None:
        """
        Fill input field.

        Args:
            selector: CSS selector
            value: Value to fill
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        logger.debug(f"Filling {selector} with: {value}")
        self.page.fill(selector, value, timeout=timeout)

    def get_text(self, selector: str, timeout: Optional[int] = None) -> str:
        """
        Get element text content.

        Args:
            selector: CSS selector
            timeout: Optional timeout in milliseconds

        Returns:
            Text content
        """
        timeout = timeout or self.timeout
        locator = self.page.locator(selector)
        locator.wait_for(state="visible", timeout=timeout)
        text = locator.text_content()
        logger.debug(f"Got text from {selector}: {text}")
        return text or ""

    def get_attribute(
        self, selector: str, attribute: str, timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Get element attribute value.

        Args:
            selector: CSS selector
            attribute: Attribute name
            timeout: Optional timeout in milliseconds

        Returns:
            Attribute value or None
        """
        timeout = timeout or self.timeout
        locator = self.page.locator(selector)
        locator.wait_for(state="attached", timeout=timeout)
        value = locator.get_attribute(attribute)
        logger.debug(f"Got attribute {attribute} from {selector}: {value}")
        return value

    def is_visible(self, selector: str, timeout: int = 5000) -> bool:
        """
        Check if element is visible.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds

        Returns:
            True if visible, False otherwise
        """
        try:
            locator = self.page.locator(selector)
            locator.wait_for(state="visible", timeout=timeout)
            return True
        except Exception:
            return False

    def is_hidden(self, selector: str, timeout: int = 5000) -> bool:
        """
        Check if element is hidden.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds

        Returns:
            True if hidden, False otherwise
        """
        try:
            locator = self.page.locator(selector)
            locator.wait_for(state="hidden", timeout=timeout)
            return True
        except Exception:
            return False

    def count_elements(self, selector: str) -> int:
        """
        Count matching elements.

        Args:
            selector: CSS selector

        Returns:
            Number of matching elements
        """
        count = self.page.locator(selector).count()
        logger.debug(f"Found {count} elements matching: {selector}")
        return count

    def get_all_text(self, selector: str) -> List[str]:
        """
        Get text from all matching elements.

        Args:
            selector: CSS selector

        Returns:
            List of text content
        """
        locators = self.page.locator(selector).all()
        texts = [loc.text_content() or "" for loc in locators]
        logger.debug(f"Got {len(texts)} text values from: {selector}")
        return texts

    def wait_for_text(
        self, selector: str, text: str, timeout: Optional[int] = None
    ) -> None:
        """
        Wait for element to contain specific text.

        Args:
            selector: CSS selector
            text: Expected text
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        locator = self.page.locator(selector)
        expect(locator).to_contain_text(text, timeout=timeout)
        logger.debug(f"Element {selector} contains text: {text}")

    def wait_for_url(self, pattern: str, timeout: Optional[int] = None) -> None:
        """
        Wait for URL to match pattern.

        Args:
            pattern: URL pattern (regex)
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        self.page.wait_for_url(pattern, timeout=timeout)
        logger.debug(f"URL matches pattern: {pattern}")

    def screenshot(self, path: str, full_page: bool = False) -> None:
        """
        Take screenshot.

        Args:
            path: File path to save screenshot
            full_page: Capture full scrollable page
        """
        self.page.screenshot(path=path, full_page=full_page)
        logger.info(f"Screenshot saved: {path}")

    def scroll_to(self, selector: str) -> None:
        """
        Scroll element into view.

        Args:
            selector: CSS selector
        """
        locator = self.page.locator(selector)
        locator.scroll_into_view_if_needed()
        logger.debug(f"Scrolled to: {selector}")

    def hover(self, selector: str, timeout: Optional[int] = None) -> None:
        """
        Hover over element.

        Args:
            selector: CSS selector
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        self.page.hover(selector, timeout=timeout)
        logger.debug(f"Hovered over: {selector}")

    def select_option(
        self, selector: str, value: str, timeout: Optional[int] = None
    ) -> None:
        """
        Select dropdown option.

        Args:
            selector: CSS selector for select element
            value: Option value to select
            timeout: Optional timeout in milliseconds
        """
        timeout = timeout or self.timeout
        self.page.select_option(selector, value, timeout=timeout)
        logger.debug(f"Selected option {value} in: {selector}")

    def press_key(self, key: str) -> None:
        """
        Press keyboard key.

        Args:
            key: Key name (e.g., 'Enter', 'Escape')
        """
        self.page.keyboard.press(key)
        logger.debug(f"Pressed key: {key}")

    def reload(self) -> None:
        """Reload current page."""
        logger.info("Reloading page")
        self.page.reload(wait_until="networkidle", timeout=self.timeout)

    def get_current_url(self) -> str:
        """
        Get current page URL.

        Returns:
            Current URL
        """
        return self.page.url

    def get_title(self) -> str:
        """
        Get page title.

        Returns:
            Page title
        """
        return self.page.title()
