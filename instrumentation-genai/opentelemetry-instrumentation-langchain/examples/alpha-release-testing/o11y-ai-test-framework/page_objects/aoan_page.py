"""
AOAN (Agent Observability & Analytics Navigator) Page Object.

Provides methods for interacting with the Agent Observability UI.
"""

from typing import List, Dict, Optional
from playwright.sync_api import Page
from page_objects.base_page import BasePage
from core.logger import get_logger


logger = get_logger(__name__)


class AOANPage(BasePage):
    """
    Page object for Agent Observability & Analytics Navigator.
    """
    
    # Selectors
    AGENT_LIST_TABLE = '[data-testid="agent-list-table"]'
    AGENT_ROW = '[data-testid="agent-row"]'
    AGENT_NAME_CELL = '[data-testid="agent-name"]'
    AGENT_STATUS_CELL = '[data-testid="agent-status"]'
    AGENT_TYPE_CELL = '[data-testid="agent-type"]'
    SEARCH_INPUT = '[data-testid="agent-search"]'
    FILTER_DROPDOWN = '[data-testid="filter-dropdown"]'
    REFRESH_BUTTON = '[data-testid="refresh-button"]'
    AGENT_DETAIL_LINK = '[data-testid="agent-detail-link"]'
    TRACE_COUNT_CELL = '[data-testid="trace-count"]'
    ERROR_COUNT_CELL = '[data-testid="error-count"]'
    LAST_SEEN_CELL = '[data-testid="last-seen"]'
    
    def __init__(self, page: Page, base_url: str):
        """
        Initialize AOAN page.
        
        Args:
            page: Playwright page instance
            base_url: Application base URL
        """
        super().__init__(page, base_url)
        self.path = "/apm/agents"
    
    def navigate(self) -> None:
        """Navigate to AOAN page."""
        logger.info("Navigating to AOAN page")
        self.navigate_to(self.path)
        self.wait_for_page_load()
    
    def wait_for_page_load(self) -> None:
        """Wait for AOAN page to fully load."""
        logger.debug("Waiting for AOAN page to load")
        self.wait_for_selector(self.AGENT_LIST_TABLE, timeout=30000)
    
    def get_agent_count(self) -> int:
        """
        Get total number of agents displayed.
        
        Returns:
            Number of agents
        """
        count = self.count_elements(self.AGENT_ROW)
        logger.info(f"Found {count} agents")
        return count
    
    def get_agent_names(self) -> List[str]:
        """
        Get list of all agent names.
        
        Returns:
            List of agent names
        """
        names = self.get_all_text(self.AGENT_NAME_CELL)
        logger.info(f"Retrieved {len(names)} agent names")
        return names
    
    def search_agent(self, search_term: str) -> None:
        """
        Search for agent by name.
        
        Args:
            search_term: Search term
        """
        logger.info(f"Searching for agent: {search_term}")
        self.fill(self.SEARCH_INPUT, search_term)
        self.press_key("Enter")
        self.wait_for_load()
    
    def click_agent(self, agent_name: str) -> None:
        """
        Click on specific agent to view details.
        
        Args:
            agent_name: Name of agent to click
        """
        logger.info(f"Clicking agent: {agent_name}")
        selector = f'{self.AGENT_NAME_CELL}:has-text("{agent_name}")'
        self.click(selector)
    
    def get_agent_status(self, agent_name: str) -> str:
        """
        Get status of specific agent.
        
        Args:
            agent_name: Agent name
        
        Returns:
            Agent status (e.g., 'active', 'inactive')
        """
        row_selector = f'{self.AGENT_ROW}:has({self.AGENT_NAME_CELL}:has-text("{agent_name}"))'
        status_selector = f'{row_selector} {self.AGENT_STATUS_CELL}'
        status = self.get_text(status_selector)
        logger.info(f"Agent {agent_name} status: {status}")
        return status
    
    def get_agent_type(self, agent_name: str) -> str:
        """
        Get type of specific agent.
        
        Args:
            agent_name: Agent name
        
        Returns:
            Agent type (e.g., 'langchain', 'langgraph')
        """
        row_selector = f'{self.AGENT_ROW}:has({self.AGENT_NAME_CELL}:has-text("{agent_name}"))'
        type_selector = f'{row_selector} {self.AGENT_TYPE_CELL}'
        agent_type = self.get_text(type_selector)
        logger.info(f"Agent {agent_name} type: {agent_type}")
        return agent_type
    
    def get_agent_trace_count(self, agent_name: str) -> int:
        """
        Get trace count for specific agent.
        
        Args:
            agent_name: Agent name
        
        Returns:
            Number of traces
        """
        row_selector = f'{self.AGENT_ROW}:has({self.AGENT_NAME_CELL}:has-text("{agent_name}"))'
        count_selector = f'{row_selector} {self.TRACE_COUNT_CELL}'
        count_text = self.get_text(count_selector)
        count = int(count_text.replace(",", ""))
        logger.info(f"Agent {agent_name} trace count: {count}")
        return count
    
    def get_agent_error_count(self, agent_name: str) -> int:
        """
        Get error count for specific agent.
        
        Args:
            agent_name: Agent name
        
        Returns:
            Number of errors
        """
        row_selector = f'{self.AGENT_ROW}:has({self.AGENT_NAME_CELL}:has-text("{agent_name}"))'
        error_selector = f'{row_selector} {self.ERROR_COUNT_CELL}'
        error_text = self.get_text(error_selector)
        count = int(error_text.replace(",", ""))
        logger.info(f"Agent {agent_name} error count: {count}")
        return count
    
    def is_agent_visible(self, agent_name: str) -> bool:
        """
        Check if agent is visible in the list.
        
        Args:
            agent_name: Agent name
        
        Returns:
            True if visible, False otherwise
        """
        selector = f'{self.AGENT_NAME_CELL}:has-text("{agent_name}")'
        visible = self.is_visible(selector, timeout=5000)
        logger.info(f"Agent {agent_name} visible: {visible}")
        return visible
    
    def refresh_agent_list(self) -> None:
        """Refresh the agent list."""
        logger.info("Refreshing agent list")
        self.click(self.REFRESH_BUTTON)
        self.wait_for_load()
    
    def apply_filter(self, filter_value: str) -> None:
        """
        Apply filter to agent list.
        
        Args:
            filter_value: Filter value to apply
        """
        logger.info(f"Applying filter: {filter_value}")
        self.click(self.FILTER_DROPDOWN)
        filter_option = f'[data-value="{filter_value}"]'
        self.click(filter_option)
        self.wait_for_load()
    
    def get_agent_details(self, agent_name: str) -> Dict[str, str]:
        """
        Get all details for specific agent.
        
        Args:
            agent_name: Agent name
        
        Returns:
            Dictionary with agent details
        """
        logger.info(f"Getting details for agent: {agent_name}")
        
        return {
            "name": agent_name,
            "status": self.get_agent_status(agent_name),
            "type": self.get_agent_type(agent_name),
            "trace_count": str(self.get_agent_trace_count(agent_name)),
            "error_count": str(self.get_agent_error_count(agent_name))
        }
    
    def wait_for_agent_to_appear(self, agent_name: str, timeout: int = 60000) -> bool:
        """
        Wait for specific agent to appear in list.
        
        Args:
            agent_name: Agent name
            timeout: Timeout in milliseconds
        
        Returns:
            True if agent appears, False otherwise
        """
        logger.info(f"Waiting for agent to appear: {agent_name}")
        selector = f'{self.AGENT_NAME_CELL}:has-text("{agent_name}")'
        
        try:
            self.wait_for_selector(selector, timeout=timeout)
            logger.info(f"Agent appeared: {agent_name}")
            return True
        except Exception as e:
            logger.warning(f"Agent did not appear: {agent_name} - {e}")
            return False
    
    def sort_by_column(self, column_name: str) -> None:
        """
        Sort agent list by column.
        
        Args:
            column_name: Column name to sort by
        """
        logger.info(f"Sorting by column: {column_name}")
        header_selector = f'[data-testid="{column_name}-header"]'
        self.click(header_selector)
        self.wait_for_load()
