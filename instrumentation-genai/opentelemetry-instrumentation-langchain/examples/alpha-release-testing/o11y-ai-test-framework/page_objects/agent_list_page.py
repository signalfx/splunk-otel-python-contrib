"""
Agent List Page Object.

Provides methods for interacting with agent list view.
"""

from typing import List, Dict
from playwright.sync_api import Page
from page_objects.base_page import BasePage
from core.logger import get_logger


logger = get_logger(__name__)


class AgentListPage(BasePage):
    """
    Page object for Agent List view.
    """

    # Selectors
    PAGE_HEADER = '[data-testid="agents-header"]'
    AGENT_CARD = '[data-testid="agent-card"]'
    AGENT_NAME = '[data-testid="agent-card-name"]'
    AGENT_DESCRIPTION = '[data-testid="agent-description"]'
    AGENT_STATUS_BADGE = '[data-testid="status-badge"]'
    AGENT_METRICS = '[data-testid="agent-metrics"]'
    CREATE_AGENT_BUTTON = '[data-testid="create-agent-button"]'
    SEARCH_BAR = '[data-testid="search-bar"]'
    FILTER_BUTTON = '[data-testid="filter-button"]'
    SORT_DROPDOWN = '[data-testid="sort-dropdown"]'
    VIEW_TOGGLE = '[data-testid="view-toggle"]'
    GRID_VIEW_BUTTON = '[data-testid="grid-view"]'
    LIST_VIEW_BUTTON = '[data-testid="list-view"]'
    PAGINATION = '[data-testid="pagination"]'
    NEXT_PAGE_BUTTON = '[data-testid="next-page"]'
    PREV_PAGE_BUTTON = '[data-testid="prev-page"]'

    def __init__(self, page: Page, base_url: str):
        """
        Initialize agent list page.

        Args:
            page: Playwright page instance
            base_url: Application base URL
        """
        super().__init__(page, base_url)
        self.path = "/agents"

    def navigate(self) -> None:
        """Navigate to agent list page."""
        logger.info("Navigating to agent list page")
        self.navigate_to(self.path)
        self.wait_for_page_load()

    def wait_for_page_load(self) -> None:
        """Wait for agent list page to fully load."""
        logger.debug("Waiting for agent list page to load")
        self.wait_for_selector(self.PAGE_HEADER, timeout=30000)

    def get_agent_count(self) -> int:
        """
        Get number of agents displayed.

        Returns:
            Agent count
        """
        count = self.count_elements(self.AGENT_CARD)
        logger.info(f"Found {count} agents")
        return count

    def get_agent_names(self) -> List[str]:
        """
        Get list of all agent names.

        Returns:
            List of agent names
        """
        names = self.get_all_text(self.AGENT_NAME)
        logger.info(f"Retrieved {len(names)} agent names")
        return names

    def click_agent(self, agent_name: str) -> None:
        """
        Click on specific agent card.

        Args:
            agent_name: Agent name
        """
        logger.info(f"Clicking agent: {agent_name}")
        selector = f'{self.AGENT_CARD}:has({self.AGENT_NAME}:has-text("{agent_name}"))'
        self.click(selector)

    def search_agents(self, search_term: str) -> None:
        """
        Search for agents.

        Args:
            search_term: Search term
        """
        logger.info(f"Searching for: {search_term}")
        self.fill(self.SEARCH_BAR, search_term)
        self.press_key("Enter")
        self.wait_for_load()

    def click_create_agent(self) -> None:
        """Click create agent button."""
        logger.info("Clicking create agent button")
        self.click(self.CREATE_AGENT_BUTTON)

    def switch_to_grid_view(self) -> None:
        """Switch to grid view."""
        logger.info("Switching to grid view")
        self.click(self.GRID_VIEW_BUTTON)
        self.wait_for_load()

    def switch_to_list_view(self) -> None:
        """Switch to list view."""
        logger.info("Switching to list view")
        self.click(self.LIST_VIEW_BUTTON)
        self.wait_for_load()

    def is_agent_visible(self, agent_name: str) -> bool:
        """
        Check if agent is visible.

        Args:
            agent_name: Agent name

        Returns:
            True if visible, False otherwise
        """
        selector = f'{self.AGENT_NAME}:has-text("{agent_name}")'
        visible = self.is_visible(selector, timeout=5000)
        logger.info(f"Agent {agent_name} visible: {visible}")
        return visible

    def get_agent_status(self, agent_name: str) -> str:
        """
        Get status of specific agent.

        Args:
            agent_name: Agent name

        Returns:
            Agent status
        """
        card_selector = (
            f'{self.AGENT_CARD}:has({self.AGENT_NAME}:has-text("{agent_name}"))'
        )
        status_selector = f"{card_selector} {self.AGENT_STATUS_BADGE}"
        status = self.get_text(status_selector)
        logger.info(f"Agent {agent_name} status: {status}")
        return status

    def get_agent_description(self, agent_name: str) -> str:
        """
        Get description of specific agent.

        Args:
            agent_name: Agent name

        Returns:
            Agent description
        """
        card_selector = (
            f'{self.AGENT_CARD}:has({self.AGENT_NAME}:has-text("{agent_name}"))'
        )
        desc_selector = f"{card_selector} {self.AGENT_DESCRIPTION}"
        description = self.get_text(desc_selector)
        logger.info(f"Agent {agent_name} description: {description}")
        return description

    def apply_filter(self, filter_type: str) -> None:
        """
        Apply filter to agent list.

        Args:
            filter_type: Filter type (e.g., 'active', 'inactive')
        """
        logger.info(f"Applying filter: {filter_type}")
        self.click(self.FILTER_BUTTON)
        filter_option = f'[data-filter="{filter_type}"]'
        self.click(filter_option)
        self.wait_for_load()

    def sort_by(self, sort_option: str) -> None:
        """
        Sort agents by option.

        Args:
            sort_option: Sort option (e.g., 'name', 'date', 'status')
        """
        logger.info(f"Sorting by: {sort_option}")
        self.click(self.SORT_DROPDOWN)
        option_selector = f'[data-sort="{sort_option}"]'
        self.click(option_selector)
        self.wait_for_load()

    def go_to_next_page(self) -> None:
        """Navigate to next page of results."""
        logger.info("Going to next page")
        self.click(self.NEXT_PAGE_BUTTON)
        self.wait_for_load()

    def go_to_prev_page(self) -> None:
        """Navigate to previous page of results."""
        logger.info("Going to previous page")
        self.click(self.PREV_PAGE_BUTTON)
        self.wait_for_load()

    def is_pagination_visible(self) -> bool:
        """
        Check if pagination is visible.

        Returns:
            True if visible, False otherwise
        """
        visible = self.is_visible(self.PAGINATION, timeout=5000)
        logger.info(f"Pagination visible: {visible}")
        return visible

    def get_all_agent_info(self) -> List[Dict[str, str]]:
        """
        Get information for all visible agents.

        Returns:
            List of agent info dictionaries
        """
        logger.info("Getting all agent info")

        agent_names = self.get_agent_names()
        agents_info = []

        for name in agent_names:
            try:
                info = {
                    "name": name,
                    "status": self.get_agent_status(name),
                    "description": self.get_agent_description(name),
                }
                agents_info.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for agent {name}: {e}")

        logger.info(f"Retrieved info for {len(agents_info)} agents")
        return agents_info

    def wait_for_agent_to_appear(self, agent_name: str, timeout: int = 60000) -> bool:
        """
        Wait for specific agent to appear.

        Args:
            agent_name: Agent name
            timeout: Timeout in milliseconds

        Returns:
            True if agent appears, False otherwise
        """
        logger.info(f"Waiting for agent to appear: {agent_name}")
        selector = f'{self.AGENT_NAME}:has-text("{agent_name}")'

        try:
            self.wait_for_selector(selector, timeout=timeout)
            logger.info(f"Agent appeared: {agent_name}")
            return True
        except Exception as e:
            logger.warning(f"Agent did not appear: {agent_name} - {e}")
            return False

    def clear_search(self) -> None:
        """Clear search input."""
        logger.info("Clearing search")
        self.fill(self.SEARCH_BAR, "")
        self.press_key("Enter")
        self.wait_for_load()
