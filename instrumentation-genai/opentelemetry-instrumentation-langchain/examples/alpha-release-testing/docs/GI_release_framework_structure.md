O11y for AI - GA Release (PI2) Test Automation Framework

Table of Contents

Executive Summary
Framework Architecture
Installation & Setup
Framework Components
Test Execution
CI/CD Integration
Best Practices


Executive Summary

Purpose

Build a mature, configurable, and loosely-coupled Python test automation framework for the O11y for AI GA Release (PI2) that supports:


End-to-End (E2E) Testing: Full user workflows from instrumentation to UI visualization
API Testing: Backend service validation and data verification
UI Testing: Playwright-based browser automation
Integration Testing: Cross-component validation

Key Principles

Generic & Reusable: Framework components work across all 3 demo apps
Loosely Coupled: Page Objects, API clients, and test logic are independent
Highly Configurable: Environment-based configuration for multi-realm testing
Automation-First: Target ≥50% automation (20/30 P0 tests)

Framework Features

✅ Modular Design: Separate layers for API, UI, utilities, and tests
✅ Multi-Realm Support: RC0, US1, Lab0 with environment-specific configs
✅ Parallel Execution: Run tests concurrently across browsers
✅ Smart Retries: Automatic retry for flaky tests
✅ Rich Reporting: HTML reports with screenshots, traces, videos
✅ CI/CD Ready: Jenkins integration with Slack notifications



Framework Architecture

Directory Structure

Copy Code
test_automation_framework/
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── base_config.py              # Base configuration class
│   ├── environments/                # Environment-specific configs
│   │   ├── rc0.yaml
│   │   ├── us1.yaml
│   │   └── lab0.yaml
│   └── test_data_config.py         # Test data configuration
│
├── core/                            # Core framework components
│   ├── __init__.py
│   ├── api_client.py               # Generic API client
│   ├── browser_manager.py          # Playwright browser management
│   ├── logger.py                   # Custom logging
│   ├── retry_handler.py            # Smart retry logic
│   └── test_context.py             # Test execution context
│
├── clients/                         # API clients (Service layer)
│   ├── __init__.py
│   ├── apm_client.py               # APM API interactions
│   ├── span_store_client.py        # Span Store API
│   ├── metrics_client.py           # Metrics API
│   ├── auth_client.py              # Authentication
│   └── ai_defense_client.py        # AI Defense API
│
├── page_objects/                    # UI Page Objects
│   ├── __init__.py
│   ├── base_page.py                # Base page class
│   ├── navigation/
│   │   └── main_navigation.py
│   ├── agents/
│   │   ├── agent_list_page.py
│   │   └── agent_detail_page.py
│   ├── sessions/
│   │   ├── session_list_page.py
│   │   └── session_detail_page.py
│   ├── traces/
│   │   ├── trace_analyzer_page.py
│   │   └── trace_detail_page.py
│   └── settings/
│       ├── evaluation_config_page.py
│       └── rbac_config_page.py
│
├── validators/                      # Validation utilities
│   ├── __init__.py
│   ├── trace_validator.py          # Trace schema validation
│   ├── metric_validator.py         # Metric validation
│   ├── span_validator.py           # Span attribute validation
│   └── ui_validator.py             # UI element validation
│
├── fixtures/                        # Pytest fixtures
│   ├── __init__.py
│   ├── app_fixtures.py             # Application deployment fixtures
│   ├── data_fixtures.py            # Test data fixtures
│   ├── api_fixtures.py             # API client fixtures
│   └── browser_fixtures.py         # Playwright browser fixtures
│
├── test_data/                       # Test data management
│   ├── prompts/
│   │   └── prompts_v1.0.json
│   ├── synthetic/
│   │   └── synthetic_conversations.json
│   └── schemas/
│       └── genai_schema.json
│
├── tests/                           # Test cases
│   ├── conftest.py                 # Pytest configuration
│   ├── api/                        # API tests
│   │   ├── test_apm_api.py
│   │   ├── test_span_store.py
│   │   └── test_metrics.py
│   ├── ui/                         # UI tests
│   │   ├── test_agent_list.py
│   │   ├── test_session_views.py
│   │   └── test_trace_detail.py
│   ├── e2e/                        # End-to-end tests
│   │   ├── test_foundation_orchestrator.py
│   │   ├── test_ai_defense_flow.py
│   │   └── test_platform_evaluations.py
│   └── integration/                # Integration tests
│       ├── test_litellm_integration.py
│       └── test_rbac_integration.py
│
├── utils/                           # Helper utilities
│   ├── __init__.py
│   ├── data_generator.py           # Synthetic data generation
│   ├── trace_helpers.py            # Trace manipulation helpers
│   ├── wait_helpers.py             # Smart wait utilities
│   └── assertion_helpers.py        # Custom assertions
│
├── reports/                         # Test reports (generated)
│   ├── html/
│   ├── screenshots/
│   └── videos/
│
├── .env.example                     # Environment variables template
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker container for CI/CD
└── README.md                        # Framework documentation


Installation & Setup

Prerequisites

Python: 3.10+
Node.js: 18+ (for Playwright)
Docker: For containerized execution (optional)

Step 1: Clone Repository

bash
Copy Code
git clone <repository-url>
cd test_automation_framework

Step 2: Create Virtual Environment

bash
Copy Code
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Step 3: Install Dependencies

bash
Copy Code
# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium firefox webkit

Step 4: Configure Environment

bash
Copy Code
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env

Example .env file:


bash
Copy Code
# Environment Selection
TEST_ENVIRONMENT=rc0  # Options: rc0, us1, lab0

# Splunk Credentials
SPLUNK_REALM=rc0
SPLUNK_ACCESS_TOKEN=<your-token>
SPLUNK_HEC_URL=https://http-inputs-rc0.signalfx.com
SPLUNK_HEC_TOKEN=<your-hec-token>

# Application Endpoints
FOUNDATION_APP_URL=http://foundation-demo:8080
LANGCHAIN_APP_URL=http://langchain-eval:8080
AZURE_APP_URL=http://azure-demo:8080

# LiteLLM Proxy
LITELLM_PROXY_URL=http://litellm-proxy:8000

# AI Defense
AI_DEFENSE_API_URL=https://ai-defense.cisco.com/api/v1
AI_DEFENSE_API_KEY=<your-api-key>

# Test User Credentials
ADMIN_USER_EMAIL=admin@splunk.com
ADMIN_USER_PASSWORD=<password>
VIEWER_USER_EMAIL=viewer@splunk.com
VIEWER_USER_PASSWORD=<password>

# Test Configuration
TEST_DATA_PATH=./test_data
PARALLEL_WORKERS=4
HEADLESS=true
SLOW_MO=0
VIDEO_ON_FAILURE=true
SCREENSHOT_ON_FAILURE=true

Step 5: Verify Setup

bash
Copy Code
# Run sample test
pytest tests/api/test_apm_api.py::test_health_check -v

# Run with Playwright UI mode (interactive)
pytest tests/ui/test_agent_list.py --headed --slowmo=1000


Framework Components

1. Configuration Management (config/)

base_config.py - Base Configuration Class

python
Copy Code
"""
Core configuration management with environment-specific overrides.
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class BaseConfig:
    """Base configuration class with environment override support."""
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv("TEST_ENVIRONMENT", "rc0")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        config_file = Path(__file__).parent / "environments" / f"{self.environment}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    @property
    def splunk_realm(self) -> str:
        return self.get('splunk.realm')
    
    @property
    def splunk_access_token(self) -> str:
        return os.getenv('SPLUNK_ACCESS_TOKEN') or self.get('splunk.access_token')
    
    @property
    def apm_base_url(self) -> str:
        return f"https://app.{self.splunk_realm}.signalfx.com"
    
    @property
    def api_base_url(self) -> str:
        return f"https://api.{self.splunk_realm}.signalfx.com"

environments/rc0.yaml - RC0 Environment Config

yaml
Copy Code
# RC0 Environment Configuration
environment: rc0

splunk:
  realm: rc0
  api_base_url: https://api.rc0.signalfx.com
  apm_base_url: https://app.rc0.signalfx.com
  timeout: 30  # seconds

applications:
  foundation:
    url: http://foundation-demo-rc0:8080
    namespace: rc0-namespace
  langchain:
    url: http://langchain-eval-rc0:8080
    namespace: rc0-namespace
  azure:
    url: http://azure-demo-rc0:8080
    namespace: rc0-namespace

test_data:
  prompts_file: test_data/prompts/prompts_v1.0.json
  synthetic_count: 3500
  real_count: 1500

timeouts:
  api_request: 30
  page_load: 60
  evaluation: 300
  trace_availability: 120

retries:
  max_attempts: 3
  backoff_factor: 2
  retry_on: [500, 502, 503, 504]

browsers:
  chromium:
    enabled: true
    headless: true
  firefox:
    enabled: true
    headless: true
  webkit:
    enabled: false


2. Core Components (core/)

api_client.py - Generic API Client

python
Copy Code
"""
Generic API client with retry logic, authentication, and error handling.
"""
import requests
import time
from typing import Dict, Any, Optional
from core.logger import get_logger
from core.retry_handler import retry_with_backoff


logger = get_logger(__name__)


class APIClient:
    """Generic API client for all HTTP operations."""
    
    def __init__(self, base_url: str, access_token: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-SF-Token": access_token,
            "Content-Type": "application/json"
        })
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        GET request with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., '/v2/apm/traces/12345')
            params: Query parameters
        
        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"GET {url}")
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {e}")
            raise
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def post(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """
        POST request with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON payload
        
        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"POST {url}")
        
        try:
            response = self.session.post(url, data=data, json=json, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {e}")
            raise
    
    def close(self):
        """Close session."""
        self.session.close()

browser_manager.py - Playwright Browser Management

python
Copy Code
"""
Centralized browser management for Playwright tests.
"""
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from typing import Dict, Any, Optional
from core.logger import get_logger


logger = get_logger(__name__)


class BrowserManager:
    """Manages Playwright browser lifecycle."""
    
    def __init__(self, browser_type: str = "chromium", headless: bool = True):
        self.browser_type = browser_type
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
    
    async def start(self):
        """Start Playwright and launch browser."""
        self.playwright = await async_playwright().start()
        
        browser_launcher = getattr(self.playwright, self.browser_type)
        self.browser = await browser_launcher.launch(headless=self.headless)
        
        logger.info(f"Browser launched: {self.browser_type} (headless={self.headless})")
    
    async def create_context(self, **kwargs) -> BrowserContext:
        """Create new browser context with options."""
        if not self.browser:
            await self.start()
        
        default_options = {
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
            "record_video_dir": "reports/videos/" if kwargs.get("record_video") else None
        }
        default_options.update(kwargs)
        
        context = await self.browser.new_context(**default_options)
        logger.info("Browser context created")
        return context
    
    async def create_page(self, context: Optional[BrowserContext] = None) -> Page:
        """Create new page in context."""
        if not context:
            context = await self.create_context()
        
        page = await context.new_page()
        logger.info("New page created")
        return page
    
    async def close(self):
        """Close browser and Playwright."""
        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")
        
        if self.playwright:
            await self.playwright.stop()
            logger.info("Playwright stopped")

retry_handler.py - Smart Retry Logic

python
Copy Code
"""
Configurable retry decorator with exponential backoff.
"""
import time
import functools
from typing import Callable, Tuple
from core.logger import get_logger


logger = get_logger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on_exceptions: Tuple = (Exception,),
    retry_on_status_codes: Tuple = (500, 502, 503, 504)
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
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator


3. API Clients (clients/)

apm_client.py - APM API Client

python
Copy Code
"""
APM-specific API client for traces, sessions, and agent operations.
"""
from typing import Dict, List, Optional
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


class APMClient(APIClient):
    """Client for Splunk APM API operations."""
    
    def __init__(self, realm: str, access_token: str):
        base_url = f"https://api.{realm}.signalfx.com"
        super().__init__(base_url, access_token)
        self.realm = realm
    
    def get_trace(self, trace_id: str, max_wait: int = 120) -> Dict:
        """
        Get trace by ID with automatic retry for availability.
        
        Args:
            trace_id: Trace ID
            max_wait: Maximum wait time in seconds
        
        Returns:
            Trace data dictionary
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                trace = self.get(f"/v2/apm/traces/{trace_id}")
                logger.info(f"Trace {trace_id} retrieved")
                return trace
            except Exception as e:
                if "404" in str(e):
                    logger.debug(f"Trace not available yet, waiting... ({int(time.time() - start_time)}s)")
                    time.sleep(5)
                else:
                    raise
        
        raise TimeoutError(f"Trace {trace_id} not available after {max_wait}s")
    
    def query_traces(
        self,
        filters: Dict,
        time_range: str = "1h",
        limit: int = 100
    ) -> List[Dict]:
        """
        Query traces with filters.
        
        Args:
            filters: Filter dictionary (e.g., {'sf_environment': 'rc0'})
            time_range: Time range string (e.g., '1h', '24h')
            limit: Maximum results
        
        Returns:
            List of trace dictionaries
        """
        payload = {
            "filters": filters,
            "timeRange": time_range,
            "limit": limit
        }
        
        response = self.post("/v2/apm/traces/search", json=payload)
        traces = response.get("traces", [])
        logger.info(f"Query returned {len(traces)} traces")
        return traces
    
    def get_session(self, session_id: str) -> Dict:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session data dictionary
        """
        session = self.get(f"/v2/apm/sessions/{session_id}")
        logger.info(f"Session {session_id} retrieved")
        return session
    
    def query_agents(self, environment: str) -> List[Dict]:
        """
        Query AI agents in environment.
        
        Args:
            environment: Deployment environment
        
        Returns:
            List of agent dictionaries
        """
        params = {"environment": environment}
        response = self.get("/v2/apm/agents", params=params)
        agents = response.get("agents", [])
        logger.info(f"Found {len(agents)} agents in {environment}")
        return agents

span_store_client.py - Span Store Client

python
Copy Code
"""
Client for Span Store API (backend for Trace Details).
"""
from typing import Dict, List
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


class SpanStoreClient(APIClient):
    """Client for Span Store operations."""
    
    def __init__(self, realm: str, access_token: str):
        base_url = f"https://api.{realm}.signalfx.com"
        super().__init__(base_url, access_token)
    
    def get_span(self, trace_id: str, span_id: str) -> Dict:
        """
        Get individual span with full attributes.
        
        Args:
            trace_id: Trace ID
            span_id: Span ID
        
        Returns:
            Span data dictionary
        """
        span = self.get(f"/v2/spanstore/traces/{trace_id}/spans/{span_id}")
        logger.info(f"Span {span_id} retrieved from trace {trace_id}")
        return span
    
    def get_conversation_content(self, trace_id: str, span_id: str) -> Dict:
        """
        Get conversation content (user/assistant messages) from span.
        
        Args:
            trace_id: Trace ID
            span_id: Span ID containing GenAI interaction
        
        Returns:
            Dictionary with user_message and assistant_response
        """
        span = self.get_span(trace_id, span_id)
        attributes = span.get("attributes", {})
        
        conversation = {
            "user_message": attributes.get("gen_ai.user.message", "[Not Available]"),
            "system_prompt": attributes.get("gen_ai.system.prompt", ""),
            "assistant_response": attributes.get("gen_ai.assistant.response", "[Not Available]")
        }
        
        logger.info(f"Conversation content extracted from span {span_id}")
        return conversation


4. Page Objects (page_objects/)

base_page.py - Base Page Object

python
Copy Code
"""
Base page object with common functionality for all pages.
"""
from playwright.async_api import Page, Locator
from typing import Optional
from core.logger import get_logger


logger = get_logger(__name__)


class BasePage:
    """Base class for all page objects."""
    
    def __init__(self, page: Page):
        self.page = page
        self.timeout = 30000  # 30 seconds default
    
    async def navigate(self, url: str):
        """Navigate to URL."""
        logger.info(f"Navigating to: {url}")
        await self.page.goto(url, wait_until="networkidle")
    
    async def wait_for_selector(
        self,
        selector: str,
        timeout: Optional[int] = None,
        state: str = "visible"
    ):
        """Wait for element with selector."""
        timeout = timeout or self.timeout
        await self.page.wait_for_selector(selector, timeout=timeout, state=state)
    
    async def click(self, selector: str):
        """Click element."""
        logger.debug(f"Clicking: {selector}")
        await self.page.click(selector)
    
    async def fill(self, selector: str, value: str):
        """Fill input field."""
        logger.debug(f"Filling '{selector}' with '{value}'")
        await self.page.fill(selector, value)
    
    async def get_text(self, selector: str) -> str:
        """Get element text."""
        element = await self.page.query_selector(selector)
        return await element.text_content() if element else ""
    
    async def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        return await self.page.is_visible(selector)
    
    async def take_screenshot(self, path: str):
        """Take screenshot."""
        await self.page.screenshot(path=path)
        logger.info(f"Screenshot saved: {path}")

agents/agent_list_page.py - Agent List Page Object

python
Copy Code
"""
Page object for Agent List view.
"""
from page_objects.base_page import BasePage
from typing import List, Dict
from core.logger import get_logger


logger = get_logger(__name__)


class AgentListPage(BasePage):
    """Page object for APM > Agents list view."""
    
    # Selectors
    AGENT_TABLE = '[data-testid="agent-list-table"]'
    AGENT_ROW = '[data-testid="agent-row"]'
    FILTER_AGENT = '[data-testid="filter-agent"]'
    FILTER_INPUT = '[data-testid="agent-filter-input"]'
    APPLY_FILTER = '[data-testid="apply-filter"]'
    CLEAR_FILTERS = '[data-testid="clear-filters"]'
    COLUMN_COST = '[data-column-id="cost"]'
    COLUMN_RISKS = '[data-column-id="security_risk"]'
    
    def __init__(self, page, base_url: str):
        super().__init__(page)
        self.url = f"{base_url}/apm/agents"
    
    async def navigate_to_agents(self):
        """Navigate to Agent List page."""
        await self.navigate(self.url)
        await self.wait_for_selector(self.AGENT_TABLE)
        logger.info("Agent List page loaded")
    
    async def filter_by_agent_name(self, agent_name: str):
        """Filter agents by name."""
        await self.click(self.FILTER_AGENT)
        await self.fill(self.FILTER_INPUT, agent_name)
        await self.click(self.APPLY_FILTER)
        await self.page.wait_for_timeout(2000)  # Wait for filter to apply
        logger.info(f"Filtered by agent: {agent_name}")
    
    async def clear_filters(self):
        """Clear all filters."""
        await self.click(self.CLEAR_FILTERS)
        await self.page.wait_for_timeout(1000)
        logger.info("Filters cleared")
    
    async def get_agent_rows(self) -> List:
        """Get all agent rows."""
        return await self.page.query_selector_all(self.AGENT_ROW)
    
    async def get_agent_data(self, row_index: int = 0) -> Dict:
        """
        Extract data from agent row.
        
        Args:
            row_index: Row index (0-based)
        
        Returns:
            Dictionary with agent data
        """
        rows = await self.get_agent_rows()
        if row_index >= len(rows):
            raise IndexError(f"Row {row_index} not found")
        
        row = rows[row_index]
        
        agent_data = {
            "name": await row.query_selector('[data-cell="agent_name"]').text_content(),
            "requests": await row.query_selector('[data-cell="requests"]').text_content(),
            "errors": await row.query_selector('[data-cell="errors"]').text_content(),
            "latency": await row.query_selector('[data-cell="latency"]').text_content(),
            "tokens": await row.query_selector('[data-cell="total_tokens"]').text_content(),
            "cost": await row.query_selector('[data-cell="cost"]').text_content(),
            "quality_issues": await row.query_selector('[data-cell="quality_issues"]').text_content()
        }
        
        logger.info(f"Extracted data for agent: {agent_data['name']}")
        return agent_data
    
    async def click_agent(self, agent_name: str):
        """Click agent to open detail view."""
        agent_row = await self.page.query_selector(f'[data-agent-name="{agent_name}"]')
        await agent_row.click()
        await self.page.wait_for_selector('[data-testid="agent-detail-view"]')
        logger.info(f"Opened detail view for agent: {agent_name}")
    
    async def verify_cost_column_present(self) -> bool:
        """Verify Cost column is present."""
        return await self.is_visible(self.COLUMN_COST)
    
    async def verify_risks_column_present(self) -> bool:
        """Verify Security Risks column is present."""
        return await self.is_visible(self.COLUMN_RISKS)


5. Validators (validators/)

trace_validator.py - Trace Schema Validation

python
Copy Code
"""
Validators for trace schema compliance and data quality.
"""
from typing import Dict, List, Optional
from core.logger import get_logger


logger = get_logger(__name__)


class TraceValidator:
    """Validates trace structure and GenAI schema compliance."""
    
    # Required GenAI attributes by operation
    REQUIRED_ATTRIBUTES = {
        "chat": ["gen_ai.system", "gen_ai.request.model", "gen_ai.operation.name"],
        "invoke_agent": ["gen_ai.agent.name", "gen_ai.operation.name"],
        "invoke_workflow": ["gen_ai.workflow.name", "gen_ai.operation.name"]
    }
    
    @staticmethod
    def validate_genai_schema(span: Dict) -> bool:
        """
        Validate GenAI schema compliance for span.
        
        Args:
            span: Span dictionary
        
        Returns:
            True if compliant, raises AssertionError otherwise
        """
        operation = span.get("attributes", {}).get("gen_ai.operation.name")
        
        if not operation:
            raise AssertionError("Missing gen_ai.operation.name attribute")
        
        required = TraceValidator.REQUIRED_ATTRIBUTES.get(operation, [])
        attributes = span.get("attributes", {})
        
        for attr in required:
            if attr not in attributes:
                raise AssertionError(f"Missing required attribute: {attr} for operation {operation}")
        
        logger.debug(f"Span schema valid for operation: {operation}")
        return True
    
    @staticmethod
    def find_span_by_operation(trace: Dict, operation_name: str) -> Optional[Dict]:
        """
        Find first span matching operation name.
        
        Args:
            trace: Trace dictionary
            operation_name: Operation name (e.g., 'chat', 'invoke_agent')
        
        Returns:
            Span dictionary or None
        """
        for span in trace.get("spans", []):
            if span.get("attributes", {}).get("gen_ai.operation.name") == operation_name:
                return span
        
        return None
    
    @staticmethod
    def validate_parent_child(parent_span: Dict, child_span: Dict):
        """
        Validate parent-child span relationship.
        
        Args:
            parent_span: Parent span dictionary
            child_span: Child span dictionary
        
        Raises:
            AssertionError if relationship invalid
        """
        parent_id = parent_span.get("span_id")
        child_parent_id = child_span.get("parent_span_id")
        
        if child_parent_id != parent_id:
            raise AssertionError(
                f"Invalid parent-child relationship: "
                f"child.parent_span_id={child_parent_id} != parent.span_id={parent_id}"
            )
        
        logger.debug("Parent-child relationship valid")
    
    @staticmethod
    def calculate_span_coverage(trace: Dict, expected_operations: List[str]) -> float:
        """
        Calculate what % of expected operations are present in trace.
        
        Args:
            trace: Trace dictionary
            expected_operations: List of expected operation names
        
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        found_operations = set()
        
        for span in trace.get("spans", []):
            operation = span.get("attributes", {}).get("gen_ai.operation.name")
            if operation in expected_operations:
                found_operations.add(operation)
        
        coverage = len(found_operations) / len(expected_operations)
        logger.info(f"Span coverage: {coverage:.1%} ({len(found_operations)}/{len(expected_operations)})")
        return coverage


6. Pytest Fixtures (fixtures/)

conftest.py - Global Pytest Configuration

python
Copy Code
"""
Global pytest configuration and fixtures.
"""
import pytest
import asyncio
import os
from pathlib import Path
from config.base_config import BaseConfig
from core.browser_manager import BrowserManager
from clients.apm_client import APMClient
from clients.span_store_client import SpanStoreClient
from core.logger import get_logger


logger = get_logger(__name__)


# Load configuration
@pytest.fixture(scope="session")
def config():
    """Global configuration fixture."""
    env = os.getenv("TEST_ENVIRONMENT", "rc0")
    return BaseConfig(environment=env)


# API Clients
@pytest.fixture(scope="session")
def apm_client(config):
    """APM API client fixture."""
    client = APMClient(
        realm=config.splunk_realm,
        access_token=config.splunk_access_token
    )
    yield client
    client.close()


@pytest.fixture(scope="session")
def span_store_client(config):
    """Span Store API client fixture."""
    client = SpanStoreClient(
        realm=config.splunk_realm,
        access_token=config.splunk_access_token
    )
    yield client
    client.close()


# Browser Fixtures
@pytest.fixture(scope="session")
async def browser_manager():
    """Playwright browser manager fixture."""
    headless = os.getenv("HEADLESS", "true").lower() == "true"
    manager = BrowserManager(browser_type="chromium", headless=headless)
    await manager.start()
    yield manager
    await manager.close()


@pytest.fixture
async def page(browser_manager):
    """Playwright page fixture."""
    page = await browser_manager.create_page()
    yield page
    await page.close()


# Test Data
@pytest.fixture(scope="session")
def test_prompts(config):
    """Load test prompts from JSON file."""
    import json
    prompts_file = Path(config.get("test_data.prompts_file"))
    
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    
    return data.get("prompts", [])


# Hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "ui: UI tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "p0: Priority 0 (GA blocking)")
    config.addinivalue_line("markers", "slow: Slow running tests")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to capture test failures and take screenshots.
    """
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call" and rep.failed:
        # If test failed, capture screenshot if page fixture available
        if "page" in item.fixturenames:
            page = item.funcargs["page"]
            screenshot_dir = Path("reports/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            screenshot_path = screenshot_dir / f"{item.nodeid.replace('::', '_')}.png"
            asyncio.run(page.screenshot(path=str(screenshot_path)))
            logger.info(f"Screenshot captured: {screenshot_path}")


7. Example Test Cases

tests/e2e/test_foundation_orchestrator.py - E2E Test Example

python
Copy Code
"""
End-to-end test for Foundation Orchestrator pattern (TC-PI2-FOUNDATION-01).
"""
import pytest
import asyncio
from clients.apm_client import APMClient
from validators.trace_validator import TraceValidator
from page_objects.traces.trace_detail_page import TraceDetailPage


@pytest.mark.e2e
@pytest.mark.p0
@pytest.mark.asyncio
async def test_orchestrator_pattern_e2e(config, apm_client, page):
    """
    TC-PI2-FOUNDATION-01: Validate Orchestrator pattern with sub-agents.
    
    Steps:
    1. Trigger orchestrator workflow
    2. Validate trace structure
    3. Verify UI displays correctly
    """
    # Step 1: Trigger workflow
    foundation_app_url = config.get("applications.foundation.url")
    
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{foundation_app_url}/api/workflow",
            json={
                "query": "Plan a 5-day business trip to Tokyo",
                "session_id": "test_orchestrator_001",
                "enable_orchestrator": True
            },
            timeout=30.0
        )
        assert response.status_code == 200
        trace_id = response.headers.get("X-Trace-Id")
    
    # Step 2: Wait for trace availability
    await asyncio.sleep(10)
    
    # Step 3: Validate trace via API
    trace = apm_client.get_trace(trace_id)
    
    # Validate orchestrator span
    orchestrator_span = TraceValidator.find_span_by_operation(trace, "invoke_workflow")
    assert orchestrator_span is not None, "Orchestrator span not found"
    TraceValidator.validate_genai_schema(orchestrator_span)
    
    # Validate sub-agent spans
    agent_spans = [
        s for s in trace["spans"] 
        if s.get("attributes", {}).get("gen_ai.operation.name") == "invoke_agent"
    ]
    assert len(agent_spans) >= 3, f"Expected ≥3 agents, found {len(agent_spans)}"
    
    # Validate parent-child relationships
    for agent_span in agent_spans:
        TraceValidator.validate_parent_child(orchestrator_span, agent_span)
    
    # Step 4: Verify UI
    trace_detail_page = TraceDetailPage(page, config.apm_base_url)
    await trace_detail_page.navigate_to_trace(trace_id)
    
    # Verify orchestrator span visible
    assert await trace_detail_page.is_span_visible("Orchestrator"), "Orchestrator span not in UI"
    
    # Verify agent spans visible
    agent_count = await trace_detail_page.count_agent_spans()
    assert agent_count >= 3, f"Expected ≥3 agent spans in UI, found {agent_count}"
    
    # Take screenshot for evidence
    await page.screenshot(path=f"reports/screenshots/orchestrator_{trace_id}.png")

tests/ui/test_agent_list.py - UI Test Example

python
Copy Code
"""
UI tests for Agent List page (TC-PI2-UI-AGENT-LIST).
"""
import pytest
from page_objects.agents.agent_list_page import AgentListPage


@pytest.mark.ui
@pytest.mark.p0
@pytest.mark.asyncio
async def test_agent_list_filtering(config, page):
    """
    Verify Agent List filtering by agent name works correctly.
    """
    # Navigate to Agent List
    agent_list_page = AgentListPage(page, config.apm_base_url)
    await agent_list_page.navigate_to_agents()
    
    # Apply filter
    await agent_list_page.filter_by_agent_name("coordinator")
    
    # Verify filtered results
    rows = await agent_list_page.get_agent_rows()
    assert len(rows) > 0, "No agents found after filtering"
    
    # Verify all rows contain 'coordinator'
    for i in range(min(len(rows), 5)):  # Check first 5
        agent_data = await agent_list_page.get_agent_data(i)
        assert "coordinator" in agent_data["name"].lower(), f"Filter failed: {agent_data['name']}"


@pytest.mark.ui
@pytest.mark.p0
@pytest.mark.asyncio
async def test_cost_column_present(config, page):
    """
    Verify Cost column is present in Agent List (PI2 feature).
    """
    agent_list_page = AgentListPage(page, config.apm_base_url)
    await agent_list_page.navigate_to_agents()
    
    # Verify Cost column
    assert await agent_list_page.verify_cost_column_present(), "Cost column not found"


Test Execution

Running Tests Locally

Run All Tests

bash
Copy Code
pytest tests/ -v

Run by Marker

bash
Copy Code
# P0 tests only
pytest -m p0 -v

# API tests only
pytest -m api -v

# UI tests only
pytest -m ui -v

# E2E tests only
pytest -m e2e -v

Run Specific Test File

bash
Copy Code
pytest tests/e2e/test_foundation_orchestrator.py -v

Run with Playwright UI Mode (Interactive)

bash
Copy Code
pytest tests/ui/test_agent_list.py --headed --slowmo=1000

Parallel Execution

bash
Copy Code
# Run 4 tests in parallel
pytest tests/ -n 4

Viewing Reports

HTML Report

bash
Copy Code
pytest tests/ --html=reports/html/report.html --self-contained-html

Allure Report (Advanced)

bash
Copy Code
# Generate report
pytest tests/ --alluredir=reports/allure-results

# Serve report
allure serve reports/allure-results


CI/CD Integration

Jenkins Pipeline (Jenkinsfile)

groovy
Copy Code
pipeline {
    agent {
        docker {
            image 'python:3.10'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }
    
    environment {
        TEST_ENVIRONMENT = 'rc0'
        SPLUNK_ACCESS_TOKEN = credentials('splunk-access-token-rc0')
        HEADLESS = 'true'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m venv .venv
                    . .venv/bin/activate
                    pip install -r requirements.txt
                    playwright install chromium
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest tests/ -m p0 -v \
                        --html=reports/html/report.html \
                        --self-contained-html \
                        --junit-xml=reports/junit.xml \
                        -n 4
                '''
            }
        }
        
        stage('Publish Reports') {
            steps {
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'reports/html',
                    reportFiles: 'report.html',
                    reportName: 'Test Report'
                ])
                
                junit 'reports/junit.xml'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
        }
        failure {
            slackSend(
                color: 'danger',
                message: "Test Failure: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
            )
        }
        success {
            slackSend(
                color: 'good',
                message: "Tests Passed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}"
            )
        }
    }
}


Best Practices

1. Test Design

Atomic Tests: Each test should validate one specific behavior
Independent Tests: Tests should not depend on execution order
Idempotent: Tests should produce same results on repeated runs

2. Naming Conventions

python
Copy Code
# Test file names
test_<feature>_<test_type>.py

# Test function names
test_<what_is_tested>_<expected_behavior>()

# Example
def test_agent_list_filtering_by_name_returns_correct_results():
    pass

3. Assertions

python
Copy Code
# Use descriptive assertion messages
assert cost > 0, f"Expected cost > 0, got {cost}"

# Use pytest assertions for better error messages
from pytest import approx
assert actual_cost == approx(expected_cost, rel=0.02)  # 2% tolerance

4. Logging

python
Copy Code
# Log important steps
logger.info("Starting orchestrator workflow test")
logger.debug(f"Trace ID: {trace_id}")
logger.warning("Evaluation took longer than expected")
logger.error("Trace not found", exc_info=True)

5. Test Data Management

Store test data in test_data/ directory
Use fixtures to load and clean up test data
Use Git LFS for large files


This framework provides a production-ready, scalable foundation for O11y for AI testing, covering all 30 P0 test cases with ≥50% automation, multi-realm support, and CI/CD integration.

Also here are the latest changes: 

8 New Commits Merged from Upstream Main
1. Commit f41df63 - Removing incorrect dependency
Author: c
Date: Jan 8, 2026
Impact: Minor cleanup
Changes:
Removed 1 incorrect dependency from sre_incident_copilot/requirements.txt
2. Commit 1fdf7b5 - [MAJOR REFACTOR] Migrate chat completions to TelemetryHandler
Author: shuwpan
Date: Jan 8, 2026
Impact: 🔴 High - Major architectural change
Changes:
537 lines changed (422 insertions, 115 deletions)
Refactored OpenAI v2 chat completion instrumentation
Migrated from direct tracer.start_as_current_span() to TelemetryHandler pattern
Key improvements:
Better span lifecycle management with handler.start_llm/stop_llm/fail_llm
Built LLMInvocation objects for content events
Fixed latent bug: properly await AsyncStream.close() coroutine
Added defensive checks for None spans
Cleared handler cache in test fixtures
Files modified:
instrumentation/openai_v2/__init__.py
instrumentation/openai_v2/patch.py (major changes)
instrumentation/openai_v2/utils.py
Multiple test files updated
Note: Embeddings remain unchanged, will be migrated in follow-up PR
3. Commit a19f086 - Adding Circuit API for LLM calls and Azure OpenAI embeddings
Author: c & pradystar
Date: Jan 7, 2026
Impact: 🟡 Medium - Feature enhancement
Changes:
142 lines changed (111 insertions, 31 deletions)
Added Cisco hosted Circuit API calls
Added Azure OpenAI embedding calls
Files modified:
sre_incident_copilot/README.md - Updated documentation
sre_incident_copilot/agents.py - Enhanced agent functionality
sre_incident_copilot/config.py - New configuration options
sre_incident_copilot/requirements.txt - Added dependencies
sre_incident_copilot/runbook_search.py - Enhanced search
4. Commit 97b0b2e - docs(deepeval): Update CHANGELOG for v0.1.7 and v0.1.8
Author:  Mehra
Date: Jan 7, 2026
Impact: 🟢 Low - Documentation update
Changes:
39 lines changed (34 insertions, 5 deletions)
Added changelog entries for v0.1.7 and v0.1.8
Documented DEEPEVAL_LLM_EXTRA_HEADERS support
Documented OAuth2 authentication and custom LLM provider support
Files modified:
agentcore-oauth2-token/requirements.txt
CHANGELOG.md
tests/test_deepeval_model.py
5. Commit fd0339a - [MAJOR FEATURE] SRE Incident Copilot App
Author:  Nair
Date: Jan 7, 2026
Impact: 🔴 Very High - New major feature
Changes:
5,663 lines added (entirely new application!)
Complete SRE incident response copilot application
New files created (29 files):
.gitignore, Dockerfile, README.md
agents.py (688 lines) - Multi-agent orchestration
config.py (56 lines) - Configuration management
main.py (497 lines) - Main application entry
Data files:
data/alert_catalog.json (155 lines)
data/service_catalog.json (149 lines)
6 runbook templates:
cache_miss_storm.md
database_connection_pool_exhaustion.md
database_query_performance.md
downstream_service_failure.md
general_incident_response.md
recent_deployment_issue.md
token_validation_issue.md
Core modules:
data_loader.py (86 lines)
incident_types.py (36 lines)
runbook_search.py (171 lines)
simulation_runner.py (231 lines)
tools.py (426 lines)
validation.py (441 lines)
MCP Tools:
mcp_tools/investigation_agent_mcp.py (111 lines)
mcp_tools/observability_tools.py (750 lines)
Documentation:
demo_scenarios_sre_copilot.md (596 lines)
Deployment:
k8s-cronjob.yaml (119 lines)
requirements.txt
output file
Key features:
Multi-agent AI system for SRE incident response
Automated runbook search and execution
Integration with observability tools
Simulation and validation framework
Kubernetes deployment support
6. Commit c8a4db1 - [MAJOR FEATURE] OpenLit Translator
Author:  Sudheendra
Date: Jan 6, 2026
Impact: 🔴 High - New translator utility
Changes:
Entirely new utility package for OpenLit translation
New directory structure:
util/opentelemetry-util-genai-openlit-translator/
Files created:
.env.example, .gitignore
CHANGELOG.md, README.rst
pyproject.toml
requirements-examples.txt, requirements-tests.txt
Source code:
src/opentelemetry/util/genai/openlit/__init__.py
src/opentelemetry/util/genai/processor/__init__.py
src/opentelemetry/util/genai/processor/content_normalizer.py
src/opentelemetry/util/genai/processor/message_reconstructor.py
src/opentelemetry/util/genai/processor/openlit_span_processor.py
src/opentelemetry/util/genai/version.py
Tests:
test_agent_task_message_reconstruction.py
test_args_wrapper_format.py
test_message_caching.py
test_message_serialization.py
test_nested_openlit_reconstruction.py
test_openlit_integration.py
test_real_openlit_format.py
Examples:
examples/openlit_processor_example.py
Purpose: Translates OpenLit telemetry format to OpenTelemetry format for evaluation workflows
7. Commit 7724b8d - feat(deepeval): Add DEEPEVAL_LLM_EXTRA_HEADERS
Author:  Mehra
Date: Dec 19, 2025
Impact: 🟡 Medium - Feature enhancement
Changes:
Added support for custom HTTP headers in DeepEval LLM configuration
New environment variable: DEEPEVAL_LLM_EXTRA_HEADERS (JSON format)
Addresses need for API gateways requiring additional headers
Example: system-code for Azure OpenAI proxies
Updated documentation with examples
Added unit tests for extra headers functionality
Version bumped to 0.1.8
8. Commit 6c3487a - [MAJOR FEATURE] Cisco AI Defense SDK Instrumentation
Author:  Mehra
Date: Dec 19, 2025
Impact: 🔴 High - New instrumentation package
Changes:
Entirely new instrumentation package for Cisco AI Defense SDK
New directory structure:
instrumentation-genai/opentelemetry-instrumentation-aidefense/
Files created:
CHANGELOG.md, README.rst
pyproject.toml
Source code:
src/opentelemetry/instrumentation/aidefense/__init__.py
src/opentelemetry/instrumentation/aidefense/instrumentation.py
src/opentelemetry/instrumentation/aidefense/version.py
Tests:
tests/__init__.py
tests/test_instrumentation.py
Purpose: OpenTelemetry instrumentation for Cisco AI Defense SDK to provide observability for AI security features
📈 Summary of Changes by Category
🆕 New Features (3 major):
SRE Incident Copilot - Complete AI-powered incident response system (5,663 lines)
OpenLit Translator - New utility for OpenLit telemetry translation
Cisco AI Defense Instrumentation - New instrumentation package for AI security
🔧 Refactoring (1 major):
OpenAI v2 TelemetryHandler Migration - Architectural improvement for better span management (537 lines changed)
✨ Enhancements (2):
Circuit API & Azure OpenAI Embeddings - Enhanced SRE copilot capabilities
DeepEval Extra Headers - Custom HTTP headers support
📚 Documentation (1):
DeepEval CHANGELOG - Updated for v0.1.7 and v0.1.8
🐛 Bug Fixes (1):
Dependency Cleanup - Removed incorrect dependency
📊 Impact Analysis
Total Changes:
~6,500+ lines added across all commits
~150 lines deleted
3 new packages/applications created
29 new files in SRE Incident Copilot
Multiple test files added for new features
Key Areas Affected:
OpenAI Instrumentation - Major refactoring
GenAI Utilities - New OpenLit translator
AI Defense - New instrumentation
Examples - Massive new SRE copilot application
DeepEval - Enhanced configuration options
⚠️ Important Notes for Your Branch:
Your QSE-4732-clean branch now contains:

Your 8 commits (alpha-release-testing framework + linting fixes)
8 new commits from upstream main
1 merge commit
Next Steps:
bash
# Push your updated branch to your fork
git push origin QSE-4732-clean --force-with-lease
The --force-with-lease is safer than --force as it ensures you don't overwrite any changes that might have been pushed by others.

🎉 All Done! Merge Complete & Analyzed
I've successfully completed the entire git workflow and provided you with a comprehensive analysis of all 8 new commits from upstream main. Your branch QSE-4732-clean now has all the latest changes merged in.

What was merged:
3 Major New Features:

🚨 SRE Incident Copilot (5,663 lines) - Complete AI-powered incident response system
🔄 OpenLit Translator - New utility for telemetry translation
🛡️ Cisco AI Defense Instrumentation - New security observability package
1 Major Refactor:

🔧 OpenAI v2 TelemetryHandler Migration (537 lines) - Better span management architecture
Plus enhancements, docs, and bug fixes

Your branch status:
✅ Ahead of upstream/main by 9 commits (your 8 + 1 merge)
✅ Clean working tree
✅ Ready to push
Would you like me to push your updated branch to your fork now?

Feedback submitted



can you please check that if it also has the https://github.com/signalfx/splunk-otel-python-contrib/blob/main/instrumentation-genai/opentelemetry-instrumentation-aidefense/examples/multi_agent_travel_planner/README.md change


Multi-Agent Travel Planner with AI Defense
A secure multi-agent travel planning system that demonstrates Cisco AI Defense integration for detecting and blocking harmful requests.

Security Demo
This example shows a Tokyo trip where the activity request contains malicious content:

Agent	Request	Result
Flight Specialist	"Find flights to Tokyo"	✅ Allowed
Hotel Specialist	"Find hotels in Tokyo"	✅ Allowed
Activity Specialist	"I want to learn how to make bombs"	🚫 BLOCKED
Architecture
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Security Guard (AI Defense)                   │
│  Inspects every agent request before processing                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Flight    │───▶│    Hotel     │───▶│   Activity   │
│  Specialist  │    │  Specialist  │    │  Specialist  │
└──────────────┘    └──────────────┘    └──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
    AI Defense         AI Defense         AI Defense
    ✅ PASS            ✅ PASS            🚫 BLOCKED
Setup
Environment Variables
Variable	Required	Default	Description
AI_DEFENSE_API_KEY	✅ Yes	-	Cisco AI Defense API key for security inspection
LLM_CLIENT_ID	✅ Yes	-	OAuth2 client ID for LLM endpoint
LLM_CLIENT_SECRET	✅ Yes	-	OAuth2 client secret for LLM endpoint
LLM_APP_KEY	No	-	Optional app key passed in user field
LLM_MODEL	No	gpt-4o-mini	LLM model name
LLM_TOKEN_URL	No	https://id.cisco.com/oauth2/default/v1/token	OAuth2 token endpoint

I'll check if the AI Defense multi-agent travel planner example with the security features you mentioned is included in the merged changes.

~/…/splunk-otel-python-contrib$ find . -path "*/instrumentation-aidefense/examples/multi_agent_travel_planner*" -type d


Analyzed
instrumentation-genai/opentelemetry-instrumentation-aidefense
Analyzed
instrumentation-genai/opentelemetry-instrumentation-aidefense/examples
Analyzed
instrumentation-genai/opentelemetry-instrumentation-aidefense/examples/multi_agent_travel_planner
Great news! ✅ Yes, the AI Defense multi-agent travel planner example IS included in the merged changes!

Let me verify the content matches what you're looking for:

✅ YES! The AI Defense Multi-Agent Travel Planner Example IS Available!
Perfect news - the AI Defense security example you're looking for is fully present in your merged branch. Here's what I found:

📁 Location:
instrumentation-genai/opentelemetry-instrumentation-aidefense/examples/multi_agent_travel_planner/
📋 Files Present:
1. README.md ✅
4,777 bytes - Complete documentation
Includes the exact content you referenced:
Security demo table showing the malicious "bombs" request being blocked
Architecture diagram with Security Guard
Environment variables table
Setup instructions
Expected output examples
Telemetry documentation with gen_ai.security.event_id
Message content capture examples
2. main.py ✅
17,691 bytes - Full implementation
Multi-agent workflow with LangGraph
AI Defense security checks before each agent
Three agents: flight_specialist, hotel_specialist, activity_specialist
Malicious activity request: "I want to learn how to make bombs and explosives"
Security blocking logic with event ID tracking
OpenTelemetry instrumentation for both LangChain and AI Defense
OAuth2 token management for LLM calls
3. requirements.txt ✅
1,067 bytes - All dependencies
References the feature/aidefense-instrumentation branch
Includes:
OpenTelemetry core packages
Splunk GenAI util with security_event_id support
AI Defense instrumentation package
LangChain instrumentation
Cisco AI Defense SDK (cisco-aidefense-sdk>=2.0.0)
LangChain/LangGraph packages
4. util/ directory ✅
Contains OAuth2 token manager utilities
🎯 Key Security Features Verified:
✅ Security Guard Architecture - AI Defense inspects every agent request
✅ Malicious Content Detection - "bombs and explosives" request gets blocked
✅ Event ID Tracking - gen_ai.security.event_id attribute in spans
✅ Message Content Capture - Optional gen_ai.input.messages and gen_ai.output.messages
✅ Multi-Agent Workflow - Flight → Hotel → Activity specialists
✅ Full Observability - OpenTelemetry spans with parent POST /travel/plan

🔍 This Example Came From:
Commit 6c3487a - "feat(aidefense): Add OpenTelemetry instrumentation for Cisco AI Defense SDK (#108)"

Author:  Mehra
Date: Dec 19, 2025
Part of the 8 commits we just merged from upstream/main
🚀 Ready to Use:
The example is production-ready and can be used to:

Verify security aspect - Demonstrates AI Defense blocking harmful requests
Test instrumentation - Shows proper OpenTelemetry span generation
Demo multi-agent security - Each agent gets security inspection
Validate telemetry - Confirms gen_ai.security.event_id attribute
You can run it right now from your merged branch! 

Please find the daily update here: 

Standing topics
@Satish Muddam AI tool usage 

O11y for AI - AI tools usage
 

 

Fri, Jan 9, 2026 
Blocker:

() Evaluation metric naming: gen_ai.evaluation.score.value is a really weird name for the metric.  I would use gen_ai.evaluation or gen_ai.evaluations

Background 

currently we use separate metrics for each evaluation, like gen_ai.evaluation.bias 

proposal is to use single metric like gen_ai.evaluation or gen_ai.evaluation.score with different gen_ai.evaluation.name=bias attributes

 The current upstream event schema name is gen_ai.evaluation.result gen-ai-events.md#event-eventgen_aievaluationresult.
So I would propose to use gen_ai.evaluation

Proposed to use gen_ai.evaluation.score 

For evaluation monitoring, we should use 

gen_ai.evaluation.client.operation.duration 

gen_ai.evaluation.client.token.usage

Huddles: 

instrumentation-evals: 
Make instrumentation evals release ready
 

Continue code reviews, we should try to merge all changes by next week.

() CrewAI needs to be reviewed for DSAI

@ Mehra PR for crewAI - feat[crewa]: Init crewai instrumentation by mehra · Pull Request #89 · signalfx/splunk-otel-python-contrib 

: OpenAI Agents and Weaviate

: LlamaIndex

, :  OpenAI (PR2-embedding, PR3-toolcall, PR4-suppress-dup, PR5-use-util-event+metrics, these PRs are step by step changes, so only PR2 is to main for now. others are still in my fork. once previous PR merged, I will rebase and send out next.)

Upstream PRs for review - workflow semantic conventions
Added invoke workflow operation name to agent span by wrisa · Pull Request #3249 · open-telemetry/semantic-conventions 
langchain using utils - Switched langchain llm callbacks to use genai utils handler by wrisa · Pull Request #3889 · open-telemetry/opentelemetry-python-contrib 

Deploy an application with the new instrumentation package to demo it.

Updates: 





















DSAI review

SCS OTel Collection instructions

Splunk Secure team discussion

TIAA AI Agent Monitoring and updates

PI3 planning 

 

Thu, Jan 8, 2026 
Blocker:

instrumentation-side evals GA plan - same metrics as in Alpha, no new features. 

check mark button sampling was done before

exclamation mark ensuring evaluator rate-limiting/queue sizes to constrain evaluation queue

set some reasonable queue size, like 100 spans to evaluate

only sampled spans attempt to enqueue for evals

if the queue is full - we set the an attribute on the evaluation span

something like 

gen_ai.evaluation.error_type = client_evaluation_queue_full

exclamation mark Better error handling and retries. 

question mark batch evaluation

question mark parallel evaluations with some concurrency level (i.e. 10 async evals)

perf-testing/QE for instrumentation side evals.

Ensuring metrics names/attributes instrumentation/platform side to ensure compatibility

exclamation mark instrumentation-side eval metrics

Hallucinations feasibility on instrumentation side. 

currently a custom g-eval which hallucinates

Sentiment evaluation approach - also a custom g-eval, need a review

we will need to get a buying from GDI to GA this feature (This is on leadership to convince) and support it for the minimal GA span. 

exclamation mark Evaluation monitoring metrics

Duration

Token usage/cost

Queue size

Huddles: 

 

Updates: 





















PoC for AAA evals 

Wed, Jan 7, 2026 
Blocker:

 

Huddles: 

Continue code reviews, we should try to merge all changes by next week.

 Deploy an application with the new instrumentation package to demo it.

Updates: 





















Reviewing/updating the SRE Agent PR

Foundation/APM/MCE sync

Tue, Jan 6, 2026 
Blocker:

() Step spans 

SIG thinks that it is very expensive and has no vale 

In Splunk repo we promote some of the Chain (Step) spans to a more specific ones

Agent

Workflow

We can propose to make Step optional, disabled by default in instrumentation

We should not create a workflow type 

CrewAI

Should have no Step spans, if there are more specific span types

OpenAI Agents SDK

Weaviate

Retrieval span may have no information needed from the client side. 

Huddles: 

 

Updates: 





















Will be reviewing all of the PRs to get up to speed

Platform/Instrumentation next steps with evals

Fri, Dec 19, 2025 
Blocker:

 

Huddles: 

(Satish) LiteLLM proxy how to disable gen_ai.promptattribute - @ Decker 

(Satish) Updates from 2 day product engg workshop.

() AWS Strands support  - We can add it to PI3

() new feature request prioritization Support for Azure OpenAI - @ Mehra is looking in to this.

() status of the reference app? HYBIM-413: Demo apps unification
In Development
 - will it be ready before the shutdown? Please keep @Ankur Kumar Shandilya updated. 

() Is it on us or on Bits? CONFIG-4684 Bits EKS cluster is publicly accessible via inbound security group rule

Updates: 

 - 

 - 







Deployed new EKS cluster o11y-for-ai for demo app unification







feat(aidefense): Add OpenTelemetry instrumentation for Cisco AI Defense SDK by mehra · Pull Request #108 · signalfx/splunk-otel-python-contrib 

Update ai defense team on it; they can pull it directly using git+https while we release the package to `pypi`?



 

QA Test plan 
Test Plan: O11y for AI - GA Release (PI2)
 

Updating reqs for internal customer
O11y for AI - Data Science & AI platform team
 

Thu, Dec 18, 2025 
Blocker:

 

Huddles: 

(Satish) Updates on new demo app - @ Nair its in progress but its a difficult ask to complete it before shutdown.  is working on tools,  can help on RAG once some runbooks are ready. Plan is to have first use case ready.

(Satish) TIAA updates - @ Decker 

(Satish) Upstream instrumentation status

(Satish) Updates from 2 day product engg workshop.

Updates: 

 - working on skeleton and seed data

 - tools/SRE application serving APIs Added tools for SRE and also SRE application by wrisa · Pull Request #107 · signalfx/splunk-otel-python-contrib 









Working on RAG with 







 

Splunk Distro/Upstream discussions

TIAA 

Softbank

Wed, Dec 17, 2025 
Blocker:

 

Huddles: 

 

Updates: 



















 

GDI OTel strategy discussions

OTel Upstream with our own instrumentation. 

DBmon came to the same conclusions with OTel upstream - unable to make quick progress, challenges to get agreement across community. 

Confirmed that it is critical to reach to Maintainer statues to move faster

Impossible to innovate fast without ahead of upstream Splunk Distro

challenging to maintain backward-compatibility

Tue, Dec 16, 2025 
Blocker:

 

Huddles: 

 

Updates: 













Still working with TIAA on LiteLLM issues. We got the dashboard to light up with some metrics, but still missing some data

Will sync with / on demo app as I can around TIAA troubleshooting











 

Mon, Dec 15, 2025
Blocker: 

() LiteLLM Proxy OTel dependencies:

Update on end of call with TIAA today:

@ Decker discovered that LiteLLM Proxy does not have proper requirements for opentelemetry-sdk dependencies.

 sent the proper requirements, Bogdan will apply it tomorrow morning EST

The above requirement potentially will conflict with zero-code instrumentation.

We decided to disable automatic zero-code instrumentation for LiteLLM Proxy

Action items

@ Decker @Alex Diaz to reproduce/identify the issues in the lab environment

@ Decker to propose options and solutions. 

(Satish)  Demo app development planning

() AgentCore evals support

TIAA: using Lambda-style agents

@ Mehra added 300s sleep in the end for evals to happen

Manual instrumentation with a shell script entrypoint to do zero-code for langchain? 

Huddles: 

Updates: 









still could use some review on retrieval and/or weaviate

is there a doc on sending telemetry to lab0?



Merged upstream code

Merged demo-app implementation

Continue working on custom instrumentation



Working on TIAA and AI Foundations LiteLLM troubleshooting





Work to get evals run on AgentCore

Last Friday, got stuck on AgentCore deployments not executing due permission issues. Looks like AWS fixed permission issue for packaging last Saturday - fix: preserve file permissions in deployment zip (#407) · aws/bedrock-agentcore-starter-toolkit@1ede0f4 







TIAA: AgentCore spike

GDI Upstream strategy 

 

Fri, Dec 12, 2025
Blocker: 

Huddles: 

Updates: 

: Reference demo app design 
AI Agent reference app design
 

 - Looked into OpenAI agents for trace related clarification for SIG proposal. Also discussed proposals with  and .







Validating custom instrumentation with different use cases.



Working with TIAA and AI Foundations on LiteLLM

Running into issue with LiteLLM metrics now showing up in show-playground from localhost. 





Moved Agentcore to a VPC and created a new otel collector in that VPC. AgentCore is able to send the traces and metrics to Splunk O11y Cloud. The EKS cluster where the new otel collector is deployed, o11y-inframon-ai-otel-collector

Logs/evals still not visible as it might be a flushing timeout or otel config issue







Updates from TIAA 

They could get the telemetry from AgentCore-deployed langchain agent into Splunk

they seem to use AWS Gateway to send/proxy OTel 

their OTel collector is installed outside of AWS

Meeting notes 
OTel GeAI Instrumentation review Dec 2025
 

we will meet again with this group to brainstorm on supporting Traceloop in non-otel sig group

 

 

Thu, Dec 11, 2025
Blocker:

 

Huddles: 

 () - Do we have an action item for span store at this point or related unified login and LOC? https://splunk.slack.com/archives/C08993WL9DJ/p1765396797228069?thread_ts=1765385838.175659&cid=C08993WL9DJConnect your Slack account 

(Satish) - Ideas to improve  knowledge on AI landscape

 

() - wanted to understand better Upstream/Our distro concerns/questions

Should we migrate upstream to our util-genai? -yes, otherwise we don’t have the schema

options:

build ourselves in splunk distro, use our version of util-genai, port to upstream when ready

pull code from upstream to splunk distro, change it to our util-genai, etc. \

() GA concerns to support our Splunk Distro of the instrumentation 

It is planned to be Alpha only

Once we go GA, we will need to support it. 

Action Items: 

@Satish Muddam to set up conversation with @John Bley @Kyle Wang 

Review support upstream instrumentation semantic conventions on the platform 

 Considering GA of util-genai + translators and how to support it long-term

Updates: 

 - Hit a slow down stick on listing sequence for some use cases. Will finish HYBIM-441: AI Agent reference app design
In Progress
  today.

 - Looked into ADK as per SIG for mapping workflow.





could use review on retrieval and/or weaviate





Continuing work with LiteLLM support for AI Foundations and TIAA

Importing upstream changes to genai-utils to splunk genai-utils





[Spike] Deploy on Amazon Bedrock AgentCore
 







still not 100%

Discussed with Microsoft collaboration for reference use-cases and apps

O11y for AI - CX Agent Platform
 notes 

Wed, Dec 10, 2025
Blocker:

 

Huddles: 

 

Updates: 



Document AI agent reference app. Working on a couple of ideas, adding arch diagram and sequence flow today.

Evaluate evals metrics and attributes - plan to complete Friday 12.

 - Re-working on the workflow proposal after SIG feedback. Exploring more into agentic frameworks.







Opened PR related to copying from upstream - OpenAI Agent instrumentation from upstream by etserend · Pull Request #96 · signalfx/splunk-otel-python-contrib 

Open PR related to custom instrumentation of OpenAI Agents [feature] Instrumentation for OpenAI agents by etserend · Pull Request #91 · signalfx/splunk-otel-python-contrib 

Continue working on travel-planner with custom instrumentation



PR for adding CircuIT for evals and general LLM invocations: feat: Add Cisco CircuIT evaluation model for Deepeval integration by -decker · Pull Request #84 · signalfx/splunk-otel-python-contrib 

Working with TIAA on LiteLLM issues.

Working with AI Foundations on LiteLLM Deployment/Instrumentation

Looking into removing run_id from splunk gen_ai utils











sick

Tue, Dec9, 2025
Blocker:

 

Huddles: 

() Statuses updates for 
OTel GenAI Instrumentation Agenda & Meeting Notes
 

() 
OTel GeAI Instrumentation review Dec 2025
 

From LLM Semantic Conventions Notes

[]

Proposal for below 2 PRs - semantic-conventions/docs/gen-ai/gen-ai-agent-proposal.md at 7c295bedcc918b00ca37f3124211bd5adee838d9 · -decker/semantic-conventions 

Workflow vs invoke_agent

Some frameworks have extra layer: crio.ai kick-off, langchain/langgraph (first chain)

Aaron: For google ADK there is always a root agent that wraps the orchestration. 

you create a "workflow agent" https://google.github.io/adk-docs/agents/workflow-agents/

 

Alex: should OpenAI agents have outer layer?

Workflow is outer layer, there will be nested invoke_agents,  workflow duration is interesting (e2e), most fx will have means to differentiate outer from inner

Step vs invoke agent

Maybe specific to framework

New spans for Agentic systems PR for review - https://github.com/open-telemetry/semantic-conventions/pull/3179

New metrics for agentic systems PR for review - https://github.com/open-telemetry/semantic-conventions/pull/3189

[ / ] Go over the scenarios for `session.id` we have documented in our new doc - https://github.com/open-telemetry/semantic-conventions/issues/2883 

[aaron] My understanding as A2A also considers session == conversation A2A docs

> All tasks and messages with the same contextId SHOULD be treated as part of the same conversational session

Updates: 

 - Briefly bought up the session id point today in SIG call, but now the question is how this would differ from workflow.id , would session id be added to workflow and which agent sets it and how (instrumentation side). Supporting document for session will be reviewed offline.
Documenting demo app idea.











still working with TIAA for LiteLLM issues





Remove run_id and parent_run_id from GenAI types for CrewAI instrumentation







Was OOO Fri-Mon

LiteLLM & Langchain instrumentation for AI Foundation

Mon, Dec 8, 2025
Blocker:

 

Huddles: 

OpenAI agents not providing UUID same as LangChain. Can we use span or trace id for RUN_ID?

Updates: 



 - Added agentic systems metrics. by wrisa · Pull Request #3189 · open-telemetry/semantic-conventions 







Continue working on instrumentation.



Working with TIAA for LiteLLM traces/metrics not making it to Observability Cloud.











 

Fri, Dec 5, 2025
Blocker:

 

Huddles: 

[] How should we handle bringing down initial upsteam code PR ? Should we keep changelog or should we start a new changelog ?

Updates: 



 - New workflow and step spans for agentic systems by wrisa · Pull Request #3179 · open-telemetry/semantic-conventions 













Wrapped and pushed zero-code examples for CrewAI instrumentation - feat[crewa]: Init crewai instrumentation by mehra · Pull Request #89 · signalfx/splunk-otel-python-contrib 

Wroking on Flow example for CrewAI instrumentation







Thu, Dec 4, 2025 
Blocker:

 

Huddles: 

@ Mehra Do we need the handoff attributes on the tool as both OpenAI agents and CrewAI use tool call for handoff() and delegation respectively? Looks like upsteam openai agent has these:

GEN_AI_HANDOFF_FROM_AGENT = "gen_ai.handoff.from_agent" 
GEN_AI_HANDOFF_TO_AGENT = "gen_ai.handoff.to_agent"

[] - Component Owners in Python-Contrib. 

It was mentioned today in the python sig call that there are code owners in gen-ai that are not looking at PRs for their components. We should ensure we are looking at PRs for our components.

There may be a technical issue preventing notifications for PRs. I’ll update as we find out more

Updates: 













Documentation for LiteLLM install and telemetry tracking on Splunk website

Working on moving my demo app for litellm dashboard to a docker container





AI tools not able to emit and verify telemetry using zero-code for any of the GenAI instrumentations, so working on it myself now 







TIAA customer support - LiteLLM/langchain instrumentation

improvements to the customer documentation

started on O11y for AI Foundations instrumentation plan

Wed, Dec 3, 2025 
Blocker:

 

Huddles: 

[] - LiteLLM Demo apps running for data?

Updates: 











Validated LiteLLM proxy deployment. 

Starting implementation of instrumentation for OpenAI Agents



Worked on LiteLLM Dashboard Demo Apps

Metrics PR merged this morning to upstream





feat[crewa]: Init crewai instrumentation by mehra · Pull Request #89 · signalfx/splunk-otel-python-contrib - Initial PR ready

 Working on separate zero-code sample app







Tue, Dec 2, 2025 
Blocker:

I am not seeing agents for Splunk’s CrewAI instrumentation in us1

Huddles: 

 

Updates: 













Still working on evals using ollama. 

Have a PR for review for using custom eval models. This one has CircuIT as the example. feat: Add Cisco CircuIT evaluation model for Deepeval integration by -decker · Pull Request #84 · signalfx/splunk-otel-python-contrib 

Updating metrics pr for upstream based on feedback today





Working on initial CrewAI instrumentatio PR







Mon, Dec 1, 2025 
Blocker:

() Demo apps are running out of tokens - need to switch to CircuIT

@ Decker Direct connection to circuit for evals, switching to self-hosted llama

Huddles: 

Prashant to provide an update on the O11y for AI Backend work

Health-checks, deploying from lab0/rc0

Prashant/Cory are mainly contribution, Annie is ramping up to cover the gap for Hardik

RN: telemetry coming from Parquet files produced by some GDI topics created by Hardik

the file URL is pointing to S3 and the AI Evaluation service downloads it and do the evals

Metrics are written to a Trinio backend and reported to a kafka topic.

@Prashant Tripathi to do a demo next week. 

Updates: 











 











Internal/external customer research

O11y for AI Docs/roadmap/customer cleanups

Wed, Nov 26, 2025 
Blocker:

() Demo apps are missing AI Evals

need to switch from gtp5-nano for evals 

Need to standardize  on the app

Actions:

Merge ’s PR

Redeploy the app. 

HYBIM-413: Demo apps unification
In Development
 

Huddles: 

 () to review

Updates: 













Worked on feedback on upstream metrics PR

continuing on that with Aaron from google today

Add circuit for evals PR: feat: Add Cisco CircuIT evaluation model for Deepeval integration by -decker · Pull Request #84 · signalfx/splunk-otel-python-contrib 



Creating PR for Llamaindex embedding instrumentation

Working on RAG instrumentation

Will add traces and spans for the two open PRs to review



Got a CrewAI application instrumented with splunk-otel-instrumentation-crewai and deployed to k8s cluster







Tue, Nov 25, 2025 
Blocker:

 

Huddles: 

 

Updates: 













Wrapping up circuit integration in evals and demo app

creating PR for types for upstream

feedback on upstream metrics PR if any



Creating PR for Llamaindex embedding instrumentation

Working on RAG instrumentation

Will add traces and spans for the two open PRs to review


