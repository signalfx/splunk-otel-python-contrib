"""
Root conftest.py for O11y AI Test Framework.

This module provides pytest configuration and root-level fixtures
that are available to all test modules.
"""

import sys
from pathlib import Path

import pytest

# Add framework root to Python path
framework_root = Path(__file__).parent
sys.path.insert(0, str(framework_root))

# Import all fixtures to make them available to tests
from fixtures.api_fixtures import *  # noqa: F401, F403, E402
from fixtures.browser_fixtures import *  # noqa: F401, F403, E402
from fixtures.app_fixtures import *  # noqa: F401, F403, E402


def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.

    Args:
        config: Pytest config object
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "smoke: Quick smoke tests for basic functionality"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests covering full workflows")
    config.addinivalue_line("markers", "api: API-level tests")
    config.addinivalue_line("markers", "ui: UI tests using Playwright")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to execute")
    config.addinivalue_line("markers", "genai: Tests for GenAI semantic conventions")
    config.addinivalue_line("markers", "evaluation: Tests for evaluation framework")
    config.addinivalue_line(
        "markers", "ai_defense: Tests for AI Defense security features"
    )
    config.addinivalue_line(
        "markers", "aoan: Tests for Agent Observability & Analytics Navigator"
    )
    config.addinivalue_line("markers", "streaming: Tests for streaming responses")
    config.addinivalue_line("markers", "multi_agent: Tests for multi-agent workflows")


def pytest_addoption(parser):
    """
    Add custom command-line options.

    Args:
        parser: Pytest parser object
    """
    parser.addoption(
        "--env",
        action="store",
        default="rc0",
        help="Environment to run tests against (rc0, us1, prod)",
    )
    parser.addoption(
        "--realm",
        action="store",
        default=None,
        help="Splunk realm (overrides environment config)",
    )
    parser.addoption(
        "--trace-id",
        action="store",
        default=None,
        help="Actual trace ID to use for validation tests",
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser tests in headless mode",
    )
    parser.addoption(
        "--slow-mo",
        action="store",
        default=0,
        type=int,
        help="Slow down browser operations by N milliseconds",
    )
    # Video recording option (skip if already registered by pytest-playwright)
    try:
        parser.addoption(
            "--video",
            action="store",
            default="off",
            help="Record videos of browser tests",
        )
    except (ValueError, Exception):
        pass  # Option already registered by pytest-playwright or other plugin

    # Tracing option (skip if already registered by pytest-playwright)
    try:
        parser.addoption(
            "--tracing",
            action="store_true",
            default=False,
            help="Enable Playwright tracing",
        )
    except (ValueError, Exception):
        pass  # Option already registered by pytest-playwright or other plugin


@pytest.fixture(scope="session")
def env(request):
    """
    Get test environment from command line.

    Returns:
        Environment name (rc0, us1, prod)
    """
    return request.config.getoption("--env")


@pytest.fixture(scope="session")
def realm(request):
    """
    Get Splunk realm from command line (optional override).

    Returns:
        Realm name or None
    """
    return request.config.getoption("--realm")


@pytest.fixture(scope="session")
def headless(request):
    """
    Get headless mode setting for browser tests.

    Returns:
        Boolean indicating headless mode
    """
    return request.config.getoption("--headless")


@pytest.fixture(scope="session")
def slow_mo(request):
    """
    Get slow motion delay for browser tests.

    Returns:
        Delay in milliseconds
    """
    return request.config.getoption("--slow-mo")


@pytest.fixture(scope="session")
def video_enabled(request):
    """
    Check if video recording is enabled.

    Returns:
        Boolean indicating video recording
    """
    return request.config.getoption("--video")


@pytest.fixture(scope="session")
def tracing_enabled(request):
    """
    Check if Playwright tracing is enabled.

    Returns:
        Boolean indicating tracing
    """
    return request.config.getoption("--tracing")


@pytest.fixture(scope="session")
def actual_trace_id(request):
    """
    Get actual trace ID from command line for validation tests.

    Returns:
        Trace ID string or None
    """
    return request.config.getoption("--trace-id")


@pytest.fixture(scope="session")
def test_run_id():
    """
    Generate unique test run ID.

    Returns:
        Unique test run identifier
    """
    import uuid
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"test_run_{timestamp}_{uuid.uuid4().hex[:8]}"
    return run_id


@pytest.fixture(scope="session")
def test_artifacts_dir(test_run_id):
    """
    Create directory for test artifacts (screenshots, videos, traces).

    Args:
        test_run_id: Unique test run identifier

    Returns:
        Path to artifacts directory
    """
    artifacts_dir = Path(__file__).parent / "test_artifacts" / test_run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Make test result available to fixtures.

    Args:
        item: Test item
        call: Test call info

    Yields:
        Test report
    """
    outcome = yield
    rep = outcome.get_result()

    # Set attribute for each phase (setup, call, teardown)
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="function")
def test_failed(request):
    """
    Check if current test has failed.

    Args:
        request: Pytest request object

    Returns:
        Boolean indicating test failure
    """
    # Access test result from hook
    return hasattr(request.node, "rep_call") and request.node.rep_call.failed
