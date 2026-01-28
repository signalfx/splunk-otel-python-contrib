"""
Trace Detail Page Object.

Provides methods for interacting with trace detail view.
"""

from typing import List, Dict, Optional
from playwright.sync_api import Page
from page_objects.base_page import BasePage
from core.logger import get_logger


logger = get_logger(__name__)


class TraceDetailPage(BasePage):
    """
    Page object for Trace Detail view.
    """

    # Selectors
    TRACE_HEADER = '[data-testid="trace-header"]'
    TRACE_ID = '[data-testid="trace-id"]'
    TRACE_DURATION = '[data-testid="trace-duration"]'
    SPAN_LIST = '[data-testid="span-list"]'
    SPAN_ROW = '[data-testid="span-row"]'
    SPAN_NAME = '[data-testid="span-name"]'
    SPAN_DURATION = '[data-testid="span-duration"]'
    SPAN_STATUS = '[data-testid="span-status"]'
    ATTRIBUTES_TAB = '[data-testid="attributes-tab"]'
    EVENTS_TAB = '[data-testid="events-tab"]'
    LOGS_TAB = '[data-testid="logs-tab"]'
    ATTRIBUTE_ROW = '[data-testid="attribute-row"]'
    ATTRIBUTE_KEY = '[data-testid="attribute-key"]'
    ATTRIBUTE_VALUE = '[data-testid="attribute-value"]'
    GENAI_SECTION = '[data-testid="genai-section"]'
    TOKEN_USAGE_PANEL = '[data-testid="token-usage"]'
    COST_PANEL = '[data-testid="cost-panel"]'
    EVALUATION_PANEL = '[data-testid="evaluation-panel"]'
    SPAN_TREE_VIEW = '[data-testid="span-tree"]'
    EXPAND_ALL_BUTTON = '[data-testid="expand-all"]'
    COLLAPSE_ALL_BUTTON = '[data-testid="collapse-all"]'

    def __init__(self, page: Page, base_url: str):
        """
        Initialize trace detail page.

        Args:
            page: Playwright page instance
            base_url: Application base URL
        """
        super().__init__(page, base_url)

    def navigate_to_trace(self, trace_id: str) -> None:
        """
        Navigate to specific trace detail page.

        Args:
            trace_id: Trace ID
        """
        path = f"/apm/traces/{trace_id}"
        logger.info(f"Navigating to trace: {trace_id}")
        self.navigate_to(path)
        self.wait_for_page_load()

    def wait_for_page_load(self) -> None:
        """Wait for trace detail page to fully load."""
        logger.debug("Waiting for trace detail page to load")
        self.wait_for_selector(self.TRACE_HEADER, timeout=30000)
        self.wait_for_selector(self.SPAN_LIST, timeout=30000)

    def get_trace_id(self) -> str:
        """
        Get trace ID from page.

        Returns:
            Trace ID
        """
        trace_id = self.get_text(self.TRACE_ID)
        logger.info(f"Trace ID: {trace_id}")
        return trace_id

    def get_trace_duration(self) -> str:
        """
        Get trace duration.

        Returns:
            Duration string (e.g., "1.23s")
        """
        duration = self.get_text(self.TRACE_DURATION)
        logger.info(f"Trace duration: {duration}")
        return duration

    def get_span_count(self) -> int:
        """
        Get number of spans in trace.

        Returns:
            Span count
        """
        count = self.count_elements(self.SPAN_ROW)
        logger.info(f"Span count: {count}")
        return count

    def get_span_names(self) -> List[str]:
        """
        Get list of all span names.

        Returns:
            List of span names
        """
        names = self.get_all_text(self.SPAN_NAME)
        logger.info(f"Retrieved {len(names)} span names")
        return names

    def click_span(self, span_name: str) -> None:
        """
        Click on specific span to view details.

        Args:
            span_name: Span name
        """
        logger.info(f"Clicking span: {span_name}")
        selector = f'{self.SPAN_NAME}:has-text("{span_name}")'
        self.click(selector)

    def get_span_status(self, span_name: str) -> str:
        """
        Get status of specific span.

        Args:
            span_name: Span name

        Returns:
            Span status (e.g., 'OK', 'ERROR')
        """
        row_selector = f'{self.SPAN_ROW}:has({self.SPAN_NAME}:has-text("{span_name}"))'
        status_selector = f"{row_selector} {self.SPAN_STATUS}"
        status = self.get_text(status_selector)
        logger.info(f"Span {span_name} status: {status}")
        return status

    def switch_to_attributes_tab(self) -> None:
        """Switch to attributes tab."""
        logger.info("Switching to attributes tab")
        self.click(self.ATTRIBUTES_TAB)
        self.wait_for_selector(self.ATTRIBUTE_ROW)

    def switch_to_events_tab(self) -> None:
        """Switch to events tab."""
        logger.info("Switching to events tab")
        self.click(self.EVENTS_TAB)

    def switch_to_logs_tab(self) -> None:
        """Switch to logs tab."""
        logger.info("Switching to logs tab")
        self.click(self.LOGS_TAB)

    def get_attribute_value(self, attribute_key: str) -> Optional[str]:
        """
        Get value of specific attribute.

        Args:
            attribute_key: Attribute key (e.g., 'gen_ai.system')

        Returns:
            Attribute value or None if not found
        """
        logger.info(f"Getting attribute: {attribute_key}")

        try:
            row_selector = f'{self.ATTRIBUTE_ROW}:has({self.ATTRIBUTE_KEY}:has-text("{attribute_key}"))'
            value_selector = f"{row_selector} {self.ATTRIBUTE_VALUE}"
            value = self.get_text(value_selector)
            logger.info(f"Attribute {attribute_key} = {value}")
            return value
        except Exception as e:
            logger.warning(f"Attribute not found: {attribute_key} - {e}")
            return None

    def get_all_attributes(self) -> Dict[str, str]:
        """
        Get all attributes as dictionary.

        Returns:
            Dictionary of attribute key-value pairs
        """
        logger.info("Getting all attributes")

        keys = self.get_all_text(self.ATTRIBUTE_KEY)
        values = self.get_all_text(self.ATTRIBUTE_VALUE)

        attributes = dict(zip(keys, values))
        logger.info(f"Retrieved {len(attributes)} attributes")
        return attributes

    def is_genai_section_visible(self) -> bool:
        """
        Check if GenAI section is visible.

        Returns:
            True if visible, False otherwise
        """
        visible = self.is_visible(self.GENAI_SECTION, timeout=5000)
        logger.info(f"GenAI section visible: {visible}")
        return visible

    def get_token_usage(self) -> Optional[Dict[str, str]]:
        """
        Get token usage information.

        Returns:
            Dictionary with token usage data or None
        """
        logger.info("Getting token usage")

        if not self.is_visible(self.TOKEN_USAGE_PANEL, timeout=5000):
            logger.warning("Token usage panel not visible")
            return None

        # Extract token usage data (implementation depends on actual UI structure)
        return {"panel_visible": "true"}

    def get_cost_info(self) -> Optional[Dict[str, str]]:
        """
        Get cost information.

        Returns:
            Dictionary with cost data or None
        """
        logger.info("Getting cost info")

        if not self.is_visible(self.COST_PANEL, timeout=5000):
            logger.warning("Cost panel not visible")
            return None

        return {"panel_visible": "true"}

    def get_evaluation_scores(self) -> Optional[Dict[str, str]]:
        """
        Get evaluation scores.

        Returns:
            Dictionary with evaluation scores or None
        """
        logger.info("Getting evaluation scores")

        if not self.is_visible(self.EVALUATION_PANEL, timeout=5000):
            logger.warning("Evaluation panel not visible")
            return None

        return {"panel_visible": "true"}

    def expand_all_spans(self) -> None:
        """Expand all spans in tree view."""
        logger.info("Expanding all spans")
        self.click(self.EXPAND_ALL_BUTTON)

    def collapse_all_spans(self) -> None:
        """Collapse all spans in tree view."""
        logger.info("Collapsing all spans")
        self.click(self.COLLAPSE_ALL_BUTTON)

    def is_span_visible(self, span_name: str) -> bool:
        """
        Check if span is visible in the list.

        Args:
            span_name: Span name

        Returns:
            True if visible, False otherwise
        """
        selector = f'{self.SPAN_NAME}:has-text("{span_name}")'
        visible = self.is_visible(selector, timeout=5000)
        logger.info(f"Span {span_name} visible: {visible}")
        return visible

    def search_attribute(self, search_term: str) -> None:
        """
        Search for attribute.

        Args:
            search_term: Search term
        """
        logger.info(f"Searching for attribute: {search_term}")
        search_input = '[data-testid="attribute-search"]'
        self.fill(search_input, search_term)
        self.press_key("Enter")
