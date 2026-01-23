# O11y for AI - Test Automation Framework
## Implementation Guide & Technical Specifications

**Version:** 2.0  
**Date:** January 12, 2026  
**Author:** Senior AI/QE Architect  
**Status:** Production Implementation

---

## Table of Contents

1. [Implementation Overview](#implementation-overview)
2. [Directory Structure](#directory-structure)
3. [Installation & Setup](#installation--setup)
4. [Core Components Implementation](#core-components-implementation)
5. [Test Case Implementation](#test-case-implementation)
6. [CI/CD Pipeline Implementation](#cicd-pipeline-implementation)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Code Examples](#code-examples)

---

## 1. Implementation Overview

### 1.1 Implementation Timeline

```
Week 1 (Dec 16-20, 2025): Framework Setup ✅ COMPLETED
├── Directory structure created
├── Base configuration implemented
├── Core components developed
└── Git LFS initialized

Week 2 (Dec 23-27, 2025): Core Testing ✅ COMPLETED
├── Foundation app tests (12 tests)
├── API client implementation
├── Playwright setup
└── First automation milestone

Week 3 (Dec 30 - Jan 3, 2026): Platform Features ✅ COMPLETED
├── Platform evaluations tests (3 tests)
├── UI automation suite (10 tests)
├── Alerting tests (2 tests)
└── App 1 100% functional

Week 4 (Jan 6-10, 2026): Integration & RBAC 🔄 IN PROGRESS
├── RBAC tests (2 tests)
├── Traceloop validation (2 tests)
├── Multi-realm testing (1 test)
└── Regression suite execution

Week 5 (Jan 13-17, 2026): Final Validation ⏳ PLANNED
├── Final regression (44 tests)
├── Performance validation
├── Documentation finalization
└── GA sign-off (Jan 20, 2026)
```

### 1.2 Technology Implementation Stack

```python
# requirements.txt
# Core Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-xdist>=3.3.0          # Parallel execution
pytest-html>=3.2.0           # HTML reporting
pytest-timeout>=2.1.0        # Test timeouts

# UI Automation
playwright>=1.40.0
pytest-playwright>=0.4.3

# API Testing
requests>=2.31.0
httpx>=0.25.0                # Async HTTP client

# Data & Config
pyyaml>=6.0
python-dotenv>=1.0.0
faker>=20.0.0                # Synthetic data generation

# Utilities
tenacity>=8.2.0              # Retry logic
structlog>=23.2.0            # Structured logging
jsonschema>=4.20.0           # Schema validation

# Reporting
allure-pytest>=2.13.0
pytest-json-report>=1.5.0

# Development
black>=23.12.0
ruff>=0.1.8
mypy>=1.7.0
```

---

## 2. Directory Structure

### 2.1 Complete Framework Structure

```
test_automation_framework/
├── README.md                        # Framework documentation
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest configuration
├── .env.example                     # Environment variables template
├── Dockerfile                       # Container for CI/CD
├── Jenkinsfile                      # CI/CD pipeline definition
│
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── base_config.py              # Base configuration class
│   ├── environments/                # Environment-specific configs
│   │   ├── __init__.py
│   │   ├── rc0.yaml                # RC0 environment
│   │   ├── us1.yaml                # US1 production
│   │   └── lab0.yaml               # Lab0 testing
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
│   │   ├── __init__.py
│   │   └── main_navigation.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_list_page.py
│   │   └── agent_detail_page.py
│   ├── sessions/
│   │   ├── __init__.py
│   │   ├── session_list_page.py
│   │   └── session_detail_page.py
│   ├── traces/
│   │   ├── __init__.py
│   │   ├── trace_analyzer_page.py
│   │   └── trace_detail_page.py
│   └── settings/
│       ├── __init__.py
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
│   │   ├── prompts_v1.0.json       # Versioned test prompts
│   │   └── .gitattributes          # Git LFS tracking
│   ├── synthetic/
│   │   └── synthetic_conversations.json
│   └── schemas/
│       └── genai_schema.json       # GenAI schema definitions
│
├── tests/                           # Test cases
│   ├── conftest.py                 # Pytest configuration
│   ├── api/                        # API tests
│   │   ├── __init__.py
│   │   ├── test_apm_api.py
│   │   ├── test_span_store.py
│   │   └── test_metrics.py
│   ├── ui/                         # UI tests
│   │   ├── __init__.py
│   │   ├── test_agent_list.py
│   │   ├── test_session_views.py
│   │   └── test_trace_detail.py
│   ├── e2e/                        # End-to-end tests
│   │   ├── __init__.py
│   │   ├── test_foundation_orchestrator.py
│   │   ├── test_ai_defense_flow.py
│   │   └── test_platform_evaluations.py
│   └── integration/                # Integration tests
│       ├── __init__.py
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
└── reports/                         # Test reports (generated)
    ├── html/
    ├── allure-results/
    ├── screenshots/
    └── videos/
```

---

## 3. Installation & Setup

### 3.1 Prerequisites

```bash
# System Requirements
- Python 3.10 or higher
- Node.js 18+ (for Playwright)
- Docker (optional, for containerized execution)
- Git LFS (for test data versioning)

# Verify installations
python --version  # Should be 3.10+
node --version    # Should be 18+
docker --version  # Should be 20+
git lfs version   # Should be 2.0+
```

### 3.2 Step-by-Step Setup

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/signalfx/splunk-otel-python-contrib.git
cd splunk-otel-python-contrib/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/alpha-release-testing

# Initialize Git LFS
git lfs install
git lfs pull
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium firefox webkit

# Verify installation
pytest --version
playwright --version
```

#### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Example .env file:**

```bash
# Environment Selection
TEST_ENVIRONMENT=rc0  # Options: rc0, us1, lab0

# Splunk Credentials
SPLUNK_REALM=rc0
SPLUNK_ACCESS_TOKEN=your-access-token-here
SPLUNK_HEC_URL=https://http-inputs-rc0.signalfx.com
SPLUNK_HEC_TOKEN=your-hec-token-here

# Application Endpoints
FOUNDATION_APP_URL=http://foundation-demo:8080
LANGCHAIN_APP_URL=http://langchain-eval:8080
AZURE_APP_URL=http://azure-demo:8080

# LiteLLM Proxy
LITELLM_PROXY_URL=http://litellm-proxy:8000

# AI Defense
AI_DEFENSE_API_URL=https://ai-defense.cisco.com/api/v1
AI_DEFENSE_API_KEY=your-ai-defense-key

# Test User Credentials
ADMIN_USER_EMAIL=admin@splunk.com
ADMIN_USER_PASSWORD=secure-password
VIEWER_USER_EMAIL=viewer@splunk.com
VIEWER_USER_PASSWORD=secure-password

# Test Configuration
TEST_DATA_PATH=./test_data
PARALLEL_WORKERS=4
HEADLESS=true
SLOW_MO=0
VIDEO_ON_FAILURE=true
SCREENSHOT_ON_FAILURE=true
```

#### Step 5: Verify Setup

```bash
# Run health check test
pytest tests/api/test_apm_api.py::test_health_check -v

# Expected output:
# tests/api/test_apm_api.py::test_health_check PASSED [100%]
```

---

## 4. Core Components Implementation

### 4.1 Base Configuration (config/base_config.py)

```python
"""
Core configuration management with environment-specific overrides.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class BaseConfig:
    """Base configuration class with environment override support."""
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration for specified environment.
        
        Args:
            environment: Environment name (rc0, us1, lab0). 
                        Defaults to TEST_ENVIRONMENT env var.
        """
        self.environment = environment or os.getenv("TEST_ENVIRONMENT", "rc0")
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration from YAML."""
        config_file = Path(__file__).parent / "environments" / f"{self.environment}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                f"Available environments: rc0, us1, lab0"
            )
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Override Splunk credentials from env vars
        if os.getenv('SPLUNK_ACCESS_TOKEN'):
            config['splunk']['access_token'] = os.getenv('SPLUNK_ACCESS_TOKEN')
        
        if os.getenv('SPLUNK_HEC_TOKEN'):
            config['splunk']['hec_token'] = os.getenv('SPLUNK_HEC_TOKEN')
        
        # Override application URLs
        if os.getenv('FOUNDATION_APP_URL'):
            config['applications']['foundation']['url'] = os.getenv('FOUNDATION_APP_URL')
        
        return config
    
    def _validate_config(self):
        """Validate required configuration keys."""
        required_keys = [
            'splunk.realm',
            'splunk.api_base_url',
            'applications.foundation.url'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Missing required configuration: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (e.g., 'splunk.realm')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config = BaseConfig('rc0')
            >>> config.get('splunk.realm')
            'rc0'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    # Convenience properties
    @property
    def splunk_realm(self) -> str:
        """Get Splunk realm."""
        return self.get('splunk.realm')
    
    @property
    def splunk_access_token(self) -> str:
        """Get Splunk access token (from env var or config)."""
        return os.getenv('SPLUNK_ACCESS_TOKEN') or self.get('splunk.access_token')
    
    @property
    def apm_base_url(self) -> str:
        """Get APM base URL."""
        return f"https://app.{self.splunk_realm}.signalfx.com"
    
    @property
    def api_base_url(self) -> str:
        """Get API base URL."""
        return f"https://api.{self.splunk_realm}.signalfx.com"
    
    @property
    def foundation_app_url(self) -> str:
        """Get Foundation app URL."""
        return self.get('applications.foundation.url')
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BaseConfig(environment='{self.environment}', realm='{self.splunk_realm}')"
```

### 4.2 Generic API Client (core/api_client.py)

```python
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
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API (e.g., https://api.rc0.signalfx.com)
            access_token: Authentication token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-SF-Token": access_token,
            "Content-Type": "application/json",
            "User-Agent": "O11y-AI-Test-Framework/2.0"
        })
        logger.info(f"APIClient initialized for {base_url}")
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        GET request with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., '/v2/apm/traces/12345')
            params: Query parameters
        
        Returns:
            Response JSON as dictionary
        
        Raises:
            requests.exceptions.RequestException: On request failure
        
        Example:
            >>> client = APIClient('https://api.rc0.signalfx.com', 'token')
            >>> trace = client.get('/v2/apm/traces/abc123')
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"GET {url}", extra={"params": params})
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            logger.debug(f"GET {url} - Status: {response.status_code}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {url}", exc_info=True)
            raise
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def post(
        self, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        POST request with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON payload
        
        Returns:
            Response JSON as dictionary
        
        Example:
            >>> client = APIClient('https://api.rc0.signalfx.com', 'token')
            >>> result = client.post('/v2/apm/traces/search', json={'filters': {}})
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"POST {url}", extra={"json": json, "data": data})
        
        try:
            response = self.session.post(
                url, 
                data=data, 
                json=json, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.debug(f"POST {url} - Status: {response.status_code}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {url}", exc_info=True)
            raise
    
    def put(
        self, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """PUT request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"PUT {url}")
        
        try:
            response = self.session.put(url, data=data, json=json, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"PUT request failed: {url}", exc_info=True)
            raise
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info(f"DELETE {url}")
        
        try:
            response = self.session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"DELETE request failed: {url}", exc_info=True)
            raise
    
    def close(self):
        """Close session and cleanup resources."""
        self.session.close()
        logger.info("APIClient session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

### 4.3 Retry Handler (core/retry_handler.py)

```python
"""
Configurable retry decorator with exponential backoff.
"""
import time
import functools
from typing import Callable, Tuple, Type
from core.logger import get_logger


logger = get_logger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retry_on_status_codes: Tuple[int, ...] = (500, 502, 503, 504)
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
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retry_on_exceptions as e:
                    # Check if it's an HTTP exception with retryable status code
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                        if status_code not in retry_on_status_codes:
                            logger.warning(
                                f"Non-retryable status code {status_code}, not retrying"
                            )
                            raise
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries ({max_attempts}) reached for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {wait_time}s...",
                        extra={"error": str(e)}
                    )
                    time.sleep(wait_time)
            
        return wrapper
    return decorator
```

### 4.4 APM Client (clients/apm_client.py)

```python
"""
APM-specific API client for traces, sessions, and agent operations.
"""
from typing import Dict, List, Optional
import time
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


class APMClient(APIClient):
    """Client for Splunk APM API operations."""
    
    def __init__(self, realm: str, access_token: str):
        """
        Initialize APM client.
        
        Args:
            realm: Splunk realm (rc0, us1, lab0)
            access_token: Splunk access token
        """
        base_url = f"https://api.{realm}.signalfx.com"
        super().__init__(base_url, access_token)
        self.realm = realm
        logger.info(f"APMClient initialized for realm: {realm}")
    
    def get_trace(self, trace_id: str, max_wait: int = 120) -> Dict:
        """
        Get trace by ID with automatic retry for availability.
        
        Args:
            trace_id: Trace ID
            max_wait: Maximum wait time in seconds
        
        Returns:
            Trace data dictionary
        
        Raises:
            TimeoutError: If trace not available after max_wait
        
        Example:
            >>> client = APMClient('rc0', 'token')
            >>> trace = client.get_trace('abc123')
            >>> print(f"Trace has {len(trace['spans'])} spans")
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                trace = self.get(f"/v2/apm/traces/{trace_id}")
                logger.info(
                    f"Trace {trace_id} retrieved",
                    extra={"span_count": len(trace.get('spans', []))}
                )
                return trace
                
            except Exception as e:
                if "404" in str(e):
                    elapsed = int(time.time() - start_time)
                    logger.debug(
                        f"Trace not available yet, waiting... ({elapsed}s/{max_wait}s)"
                    )
                    time.sleep(5)
                else:
                    raise
        
        raise TimeoutError(
            f"Trace {trace_id} not available after {max_wait}s. "
            f"Check if telemetry is being sent correctly."
        )
    
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
        
        Example:
            >>> client = APMClient('rc0', 'token')
            >>> traces = client.query_traces(
            ...     filters={'gen_ai.operation.name': 'chat'},
            ...     time_range='1h'
            ... )
        """
        payload = {
            "filters": filters,
            "timeRange": time_range,
            "limit": limit
        }
        
        response = self.post("/v2/apm/traces/search", json=payload)
        traces = response.get("traces", [])
        
        logger.info(
            f"Query returned {len(traces)} traces",
            extra={"filters": filters, "time_range": time_range}
        )
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
    
    def query_agents(
        self, 
        environment: str, 
        time_range: str = "1h"
    ) -> List[Dict]:
        """
        Query AI agents in environment.
        
        Args:
            environment: Deployment environment
            time_range: Time range for metrics
        
        Returns:
            List of agent dictionaries
        """
        params = {
            "environment": environment,
            "timeRange": time_range
        }
        
        response = self.get("/v2/apm/agents", params=params)
        agents = response.get("agents", [])
        
        logger.info(
            f"Found {len(agents)} agents in {environment}",
            extra={"environment": environment}
        )
        return agents
```

---

## 5. Test Case Implementation

### 5.1 E2E Test Example (tests/e2e/test_foundation_orchestrator.py)

```python
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
    2. Validate trace structure via API
    3. Verify UI displays correctly
    
    Expected:
    - Orchestrator span present with gen_ai.operation.name=invoke_workflow
    - 3+ sub-agent spans with gen_ai.operation.name=invoke_agent
    - Correct parent-child hierarchy
    - UI displays workflow hierarchy
    """
    # Step 1: Trigger workflow
    foundation_app_url = config.foundation_app_url
    
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
        assert response.status_code == 200, f"Workflow trigger failed: {response.status_code}"
        trace_id = response.headers.get("X-Trace-Id")
        assert trace_id, "No trace ID in response headers"
    
    # Step 2: Wait for trace availability
    await asyncio.sleep(10)
    
    # Step 3: Validate trace via API
    trace = apm_client.get_trace(trace_id)
    
    # Validate orchestrator span
    orchestrator_span = TraceValidator.find_span_by_operation(trace, "invoke_workflow")
    assert orchestrator_span is not None, "Orchestrator span not found in trace"
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
    assert await trace_detail_page.is_span_visible("Orchestrator"), \
        "Orchestrator span not visible in UI"
    
    # Verify agent spans visible
    agent_count = await trace_detail_page.count_agent_spans()
    assert agent_count >= 3, f"Expected ≥3 agent spans in UI, found {agent_count}"
    
    # Take screenshot for evidence
    await page.screenshot(path=f"reports/screenshots/orchestrator_{trace_id}.png")
```

### 5.2 API Test Example (tests/api/test_apm_api.py)

```python
"""
API tests for APM endpoints.
"""
import pytest
from clients.apm_client import APMClient


@pytest.mark.api
@pytest.mark.p0
def test_health_check(apm_client):
    """Verify APM API is accessible."""
    # Simple health check - query agents
    agents = apm_client.query_agents(environment="test", time_range="1h")
    assert isinstance(agents, list), "Expected list of agents"


@pytest.mark.api
@pytest.mark.p0
def test_trace_retrieval(apm_client, test_trace_id):
    """Verify trace retrieval works correctly."""
    trace = apm_client.get_trace(test_trace_id)
    
    # Validate trace structure
    assert "trace_id" in trace
    assert "spans" in trace
    assert len(trace["spans"]) > 0
    
    # Validate span structure
    first_span = trace["spans"][0]
    assert "span_id" in first_span
    assert "name" in first_span
    assert "attributes" in first_span
```

### 5.3 UI Test Example (tests/ui/test_agent_list.py)

```python
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
        assert "coordinator" in agent_data["name"].lower(), \
            f"Filter failed: {agent_data['name']}"


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
    assert await agent_list_page.verify_cost_column_present(), \
        "Cost column not found in Agent List"
```

---

## 6. CI/CD Pipeline Implementation

### 6.1 Jenkinsfile

```groovy
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
        PARALLEL_WORKERS = '4'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    echo "Setting up test environment..."
                    python -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    playwright install chromium
                '''
            }
        }
        
        stage('Lint') {
            steps {
                sh '''
                    . .venv/bin/activate
                    echo "Running linters..."
                    ruff check .
                    black --check .
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    echo "Running P0 tests..."
                    pytest tests/ -m p0 -v \
                        --html=reports/html/report.html \
                        --self-contained-html \
                        --junit-xml=reports/junit.xml \
                        -n ${PARALLEL_WORKERS}
                '''
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    . .venv/bin/activate
                    echo "Generating Allure report..."
                    allure generate reports/allure-results -o reports/allure-report --clean
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
                
                allure([
                    includeProperties: false,
                    jdk: '',
                    properties: [],
                    reportBuildPolicy: 'ALWAYS',
                    results: [[path: 'reports/allure-results']]
                ])
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
            archiveArtifacts artifacts: 'reports/screenshots/**/*.png', allowEmptyArchive: true
        }
        
        failure {
            slackSend(
                color: 'danger',
                message: "❌ Test Failure: ${env.JOB_NAME} - ${env.BUILD_NUMBER}\n" +
                         "View: ${env.BUILD_URL}"
            )
        }
        
        success {
            slackSend(
                color: 'good',
                message: "✅ Tests Passed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}\n" +
                         "View: ${env.BUILD_URL}"
            )
        }
    }
}
```

---

## 7. Deployment Guide

### 7.1 Local Deployment

```bash
# 1. Start local services (if needed)
docker-compose up -d

# 2. Run tests locally
pytest tests/ -m p0 -v

# 3. View HTML report
open reports/html/report.html
```

### 7.2 CI/CD Deployment

```bash
# 1. Push to repository
git push origin feature/test-automation

# 2. Jenkins automatically triggers pipeline

# 3. View results in Jenkins UI
# Navigate to: https://jenkins.example.com/job/o11y-ai-tests/
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue: Playwright browsers not installed**
```bash
# Solution:
playwright install chromium firefox webkit
```

**Issue: Test data not found**
```bash
# Solution: Pull Git LFS files
git lfs pull
```

**Issue: API authentication fails**
```bash
# Solution: Check access token
echo $SPLUNK_ACCESS_TOKEN
# Regenerate token if expired
```

---

## 9. Best Practices

### 9.1 Test Design
- ✅ Keep tests atomic and independent
- ✅ Use descriptive test names
- ✅ Add clear assertions with messages
- ✅ Clean up test data after execution

### 9.2 Code Quality
- ✅ Follow PEP 8 style guide
- ✅ Use type hints
- ✅ Write docstrings for all functions
- ✅ Keep functions small and focused

### 9.3 Maintenance
- ✅ Review and update tests regularly
- ✅ Keep dependencies up to date
- ✅ Monitor test execution times
- ✅ Refactor duplicate code

---

## 10. Code Examples

See the complete implementation in the repository:
- `config/` - Configuration examples
- `core/` - Core component examples
- `tests/` - Test case examples
- `page_objects/` - Page object examples

---

**Document Owner:** Ankur Kumar Shandilya  
**Last Updated:** January 12, 2026  
**Review Cycle:** Monthly
