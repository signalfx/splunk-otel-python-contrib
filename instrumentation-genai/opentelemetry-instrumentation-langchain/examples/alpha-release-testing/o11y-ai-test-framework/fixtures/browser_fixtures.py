"""
Browser-related pytest fixtures using Playwright.

Provides fixtures for browser automation, page objects, and UI testing.
"""

import pytest
from playwright.sync_api import sync_playwright
from core.logger import get_logger


logger = get_logger(__name__)


@pytest.fixture(scope="session")
def playwright_instance():
    """
    Create Playwright instance for session.
    
    Yields:
        Playwright instance
    """
    with sync_playwright() as p:
        logger.info("Started Playwright instance")
        yield p
        logger.info("Stopped Playwright instance")


@pytest.fixture(scope="session")
def browser(playwright_instance, headless, slow_mo):
    """
    Launch browser for session.
    
    Args:
        playwright_instance: Playwright instance
        headless: Run in headless mode
        slow_mo: Slow down operations (ms)
    
    Yields:
        Browser instance
    """
    browser = playwright_instance.chromium.launch(
        headless=headless,
        slow_mo=slow_mo
    )
    logger.info(f"Launched browser (headless={headless}, slow_mo={slow_mo})")
    yield browser
    browser.close()
    logger.info("Closed browser")


@pytest.fixture(scope="function")
def context(browser, video_enabled, tracing_enabled, test_artifacts_dir):
    """
    Create browser context for each test.
    
    Args:
        browser: Browser instance
        video_enabled: Enable video recording
        tracing_enabled: Enable Playwright tracing
        test_artifacts_dir: Directory for artifacts
    
    Yields:
        BrowserContext instance
    """
    context_options = {
        "viewport": {"width": 1920, "height": 1080},
        "ignore_https_errors": True
    }
    
    if video_enabled:
        context_options["record_video_dir"] = str(test_artifacts_dir / "videos")
        context_options["record_video_size"] = {"width": 1920, "height": 1080}
    
    context = browser.new_context(**context_options)
    
    if tracing_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
    
    logger.info("Created browser context")
    yield context
    
    # Save trace if enabled
    if tracing_enabled:
        trace_path = test_artifacts_dir / "traces" / "trace.zip"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        context.tracing.stop(path=str(trace_path))
        logger.info(f"Saved trace to: {trace_path}")
    
    context.close()
    logger.info("Closed browser context")


@pytest.fixture(scope="function")
def page(context):
    """
    Create page for each test.
    
    Args:
        context: Browser context
    
    Yields:
        Page instance
    """
    page = context.new_page()
    logger.info("Created new page")
    yield page
    page.close()
    logger.info("Closed page")


@pytest.fixture(scope="function")
def authenticated_page(page, app_base_url, config, test_failed, test_artifacts_dir):
    """
    Create authenticated page with Splunk login.
    
    Args:
        page: Page instance
        app_base_url: Application base URL
        config: Config instance
        test_failed: Test failure status
        test_artifacts_dir: Directory for artifacts
    
    Yields:
        Authenticated page instance
    """
    # Navigate to login page
    login_url = f"{app_base_url}/login"
    page.goto(login_url, wait_until="networkidle")
    logger.info(f"Navigated to: {login_url}")
    
    # Perform login
    username = config.get("test_user.username")
    password = config.get("test_user.password")
    
    try:
        # Fill login form (adjust selectors based on actual UI)
        page.fill('input[name="username"]', username)
        page.fill('input[name="password"]', password)
        page.click('button[type="submit"]')
        
        # Wait for navigation after login
        page.wait_for_load_state("networkidle")
        logger.info("Successfully logged in")
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        screenshot_path = test_artifacts_dir / "screenshots" / "login_failure.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path))
        raise
    
    yield page
    
    # Take screenshot on test failure
    if test_failed:
        screenshot_path = test_artifacts_dir / "screenshots" / "test_failure.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot_path), full_page=True)
        logger.info(f"Saved failure screenshot to: {screenshot_path}")


@pytest.fixture(scope="function")
def aoan_page(authenticated_page, app_base_url):
    """
    Navigate to AOAN (Agent Observability & Analytics Navigator) page.
    
    Args:
        authenticated_page: Authenticated page instance
        app_base_url: Application base URL
    
    Returns:
        Page on AOAN view
    """
    aoan_url = f"{app_base_url}/apm/agents"
    authenticated_page.goto(aoan_url, wait_until="networkidle")
    logger.info(f"Navigated to AOAN: {aoan_url}")
    return authenticated_page


@pytest.fixture(scope="function")
def trace_detail_page(authenticated_page, app_base_url):
    """
    Helper to navigate to trace detail page.
    
    Args:
        authenticated_page: Authenticated page instance
        app_base_url: Application base URL
    
    Returns:
        Function to navigate to specific trace
    """
    def navigate_to_trace(trace_id: str):
        trace_url = f"{app_base_url}/apm/traces/{trace_id}"
        authenticated_page.goto(trace_url, wait_until="networkidle")
        logger.info(f"Navigated to trace: {trace_id}")
        return authenticated_page
    
    return navigate_to_trace


@pytest.fixture(scope="function")
def screenshot_on_failure(page, test_failed, test_artifacts_dir, request):
    """
    Automatically capture screenshot on test failure.
    
    Args:
        page: Page instance
        test_failed: Test failure status
        test_artifacts_dir: Directory for artifacts
        request: Pytest request object
    
    Yields:
        None (runs after test)
    """
    yield
    
    if test_failed:
        test_name = request.node.name
        screenshot_path = test_artifacts_dir / "screenshots" / f"{test_name}_failure.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info(f"Captured failure screenshot: {screenshot_path}")
        except Exception as e:
            logger.warning(f"Failed to capture screenshot: {e}")


@pytest.fixture(scope="session")
def browser_timeout(config):
    """
    Get browser operation timeout from config.
    
    Args:
        config: Config instance
    
    Returns:
        Timeout in milliseconds
    """
    return config.get("timeouts.page_load", 30) * 1000


@pytest.fixture(scope="function")
def set_browser_timeout(page, browser_timeout):
    """
    Set default timeout for browser operations.
    
    Args:
        page: Page instance
        browser_timeout: Timeout in milliseconds
    """
    page.set_default_timeout(browser_timeout)
    page.set_default_navigation_timeout(browser_timeout)
    logger.debug(f"Set browser timeout to: {browser_timeout}ms")
