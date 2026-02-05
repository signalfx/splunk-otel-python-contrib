"""
API-related pytest fixtures.

Provides fixtures for API clients, configuration, and API test utilities.
"""

import pytest
from config.base_config import BaseConfig
from clients.apm_client import APMClient
from core.api_client import APIClient
from core.logger import get_logger


logger = get_logger(__name__)


@pytest.fixture(scope="session")
def config(env, realm):
    """
    Load configuration for test environment.

    Args:
        env: Environment name (rc0, us1, prod)
        realm: Optional realm override

    Returns:
        BaseConfig instance
    """
    cfg = BaseConfig(environment=env)

    # Override realm if specified
    if realm:
        cfg.config["splunk"]["realm"] = realm
        logger.info(f"Overriding realm to: {realm}")

    logger.info(f"Loaded config for environment: {env}")
    return cfg


@pytest.fixture(scope="session")
def splunk_access_token(config):
    """
    Get Splunk access token from config.

    Args:
        config: Config instance

    Returns:
        Access token string
    """
    token = config.get("splunk.access_token")
    assert token, "Splunk access token not configured"
    return token


@pytest.fixture(scope="session")
def splunk_realm(config):
    """
    Get Splunk realm from config.

    Args:
        config: Config instance

    Returns:
        Realm string
    """
    realm = config.get("splunk.realm")
    assert realm, "Splunk realm not configured"
    return realm


@pytest.fixture(scope="session")
def apm_base_url(config):
    """
    Get APM API base URL.

    Args:
        config: Config instance

    Returns:
        APM API base URL
    """
    return config.apm_api_url


@pytest.fixture(scope="session")
def app_base_url(config):
    """
    Get O11y application base URL.

    Args:
        config: Config instance

    Returns:
        Application base URL
    """
    return config.app_url


@pytest.fixture(scope="session")
def api_client(apm_base_url, splunk_access_token):
    """
    Create generic API client.

    Args:
        apm_base_url: APM API base URL
        splunk_access_token: Access token

    Returns:
        APIClient instance
    """
    client = APIClient(
        base_url=apm_base_url, headers={"X-SF-TOKEN": splunk_access_token}
    )
    logger.info(f"Created API client for: {apm_base_url}")
    return client


@pytest.fixture(scope="session")
def apm_client(splunk_realm, config):
    """
    Create APM client for trace/span operations using session authentication.

    Args:
        splunk_realm: Splunk realm (rc0, us1, lab0)
        config: Config instance with credentials

    Returns:
        APMClient instance
    """
    # Get credentials from config or environment
    import os

    email = os.getenv("O11Y_CLOUD_USERNAME") or config.get("test_users.admin.email")
    password = os.getenv("O11Y_CLOUD_PASSWORD") or config.get(
        "test_users.admin.password"
    )

    if not email or not password:
        raise ValueError(
            "O11y Cloud credentials not configured. "
            "Set O11Y_CLOUD_USERNAME and O11Y_CLOUD_PASSWORD environment variables."
        )

    # Get target org from environment (default to qaregression for trace queries)
    target_org = os.getenv("O11Y_TARGET_ORG", "qaregression")

    client = APMClient(
        realm=splunk_realm, email=email, password=password, target_org=target_org
    )
    logger.info(f"Created APM client for realm: {splunk_realm}, org: {target_org}")
    return client


@pytest.fixture(scope="function")
def trace_id_list():
    """
    Provide list to collect trace IDs during test.

    Returns:
        Empty list for trace IDs
    """
    return []


@pytest.fixture(scope="function")
def session_id_list():
    """
    Provide list to collect session IDs during test.

    Returns:
        Empty list for session IDs
    """
    return []


@pytest.fixture(scope="function")
def cleanup_traces(apm_client, trace_id_list, request):
    """
    Cleanup fixture to delete test traces after test.

    Args:
        apm_client: APM client instance
        trace_id_list: List of trace IDs to cleanup
        request: Pytest request object

    Yields:
        None (fixture runs cleanup after test)
    """
    yield

    # Cleanup only if test passed (optional behavior)
    if not request.node.rep_call.failed and trace_id_list:
        logger.info(f"Cleaning up {len(trace_id_list)} test traces")
        for trace_id in trace_id_list:
            try:
                # Note: Implement delete_trace in APMClient if needed
                logger.debug(f"Would cleanup trace: {trace_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup trace {trace_id}: {e}")


@pytest.fixture(scope="session")
def api_timeout(config):
    """
    Get API timeout from config.

    Args:
        config: Config instance

    Returns:
        Timeout in seconds
    """
    return config.get("timeouts.api_request", 30)


@pytest.fixture(scope="session")
def trace_wait_timeout(config):
    """
    Get trace availability wait timeout.

    Args:
        config: Config instance

    Returns:
        Timeout in seconds
    """
    return config.get("timeouts.trace_availability", 120)


@pytest.fixture(scope="session")
def max_retries(config):
    """
    Get max retry count from config.

    Args:
        config: Config instance

    Returns:
        Max retry count
    """
    return config.get("retry.max_attempts", 3)
