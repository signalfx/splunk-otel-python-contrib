"""
Application deployment and management fixtures.

Provides fixtures for deploying and managing test applications,
including LangChain, LangGraph, and LiteLLM apps.
"""

import pytest
import subprocess
import time
from pathlib import Path
from core.logger import get_logger
from utils.data_generator import DataGenerator


logger = get_logger(__name__)


@pytest.fixture(scope="session")
def test_apps_dir():
    """
    Get test applications directory.

    Returns:
        Path to test apps directory
    """
    framework_root = Path(__file__).parent.parent
    apps_dir = framework_root.parent.parent / "tests" / "apps"
    return apps_dir


@pytest.fixture(scope="session")
def litellm_config(config):
    """
    Get LiteLLM configuration.

    Args:
        config: Config instance

    Returns:
        LiteLLM config dictionary
    """
    return {
        "base_url": config.get("litellm.base_url"),
        "api_key": config.get("litellm.api_key"),
        "model": config.get("litellm.model", "gpt-3.5-turbo"),
    }


@pytest.fixture(scope="session")
def ai_defense_config(config):
    """
    Get AI Defense configuration.

    Args:
        config: Config instance

    Returns:
        AI Defense config dictionary
    """
    return {
        "enabled": config.get("ai_defense.enabled", False),
        "base_url": config.get("ai_defense.base_url"),
        "client_id": config.get("ai_defense.client_id"),
        "client_secret": config.get("ai_defense.client_secret"),
    }


@pytest.fixture(scope="function")
def test_session_id():
    """
    Generate unique session ID for test.

    Returns:
        Session ID string
    """
    return DataGenerator.generate_session_id()


@pytest.fixture(scope="function")
def test_prompts(count=5):
    """
    Generate test prompts.

    Args:
        count: Number of prompts to generate

    Returns:
        List of prompt strings
    """
    return DataGenerator.generate_prompts(count)


@pytest.fixture(scope="function")
def test_conversation(turns=3):
    """
    Generate test conversation.

    Args:
        turns: Number of conversation turns

    Returns:
        List of message dictionaries
    """
    return DataGenerator.generate_conversation(turns)


@pytest.fixture(scope="session")
def deployed_app(test_apps_dir, config, request):
    """
    Deploy test application for session.

    Args:
        test_apps_dir: Test apps directory
        config: Config instance
        request: Pytest request object

    Yields:
        Deployed app info dictionary
    """
    # Get app name from marker or default
    marker = request.node.get_closest_marker("app")
    app_name = marker.args[0] if marker else "langchain_evaluation_app"

    app_path = test_apps_dir / f"{app_name}.py"

    if not app_path.exists():
        logger.warning(f"App not found: {app_path}")
        yield {"name": app_name, "running": False}
        return

    # Start app as subprocess
    logger.info(f"Starting app: {app_name}")

    env = {
        "SPLUNK_ACCESS_TOKEN": config.get("splunk.access_token"),
        "SPLUNK_REALM": config.get("splunk.realm"),
        "OTEL_SERVICE_NAME": f"test_{app_name}",
        "OTEL_RESOURCE_ATTRIBUTES": f"deployment.environment={config.environment}",
    }

    process = subprocess.Popen(
        ["python", str(app_path)],
        env={**subprocess.os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for app to start
    time.sleep(5)

    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"App failed to start: {stderr.decode()}")
        yield {"name": app_name, "running": False, "error": stderr.decode()}
        return

    logger.info(f"App started: {app_name} (PID: {process.pid})")

    yield {"name": app_name, "running": True, "pid": process.pid, "process": process}

    # Cleanup
    logger.info(f"Stopping app: {app_name}")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning(f"Force killing app: {app_name}")
        process.kill()

    logger.info(f"App stopped: {app_name}")


@pytest.fixture(scope="function")
def app_execution_context(deployed_app, test_session_id, splunk_access_token):
    """
    Provide execution context for app tests.

    Args:
        deployed_app: Deployed app info
        test_session_id: Test session ID
        splunk_access_token: Splunk access token

    Returns:
        Execution context dictionary
    """
    return {
        "app_name": deployed_app.get("name"),
        "session_id": test_session_id,
        "access_token": splunk_access_token,
        "running": deployed_app.get("running", False),
    }


@pytest.fixture(scope="session")
def openai_api_key(config):
    """
    Get OpenAI API key from config.

    Args:
        config: Config instance

    Returns:
        OpenAI API key
    """
    api_key = config.get("openai.api_key")
    if not api_key:
        pytest.skip("OpenAI API key not configured")
    return api_key


@pytest.fixture(scope="session")
def anthropic_api_key(config):
    """
    Get Anthropic API key from config.

    Args:
        config: Config instance

    Returns:
        Anthropic API key
    """
    api_key = config.get("anthropic.api_key")
    if not api_key:
        pytest.skip("Anthropic API key not configured")
    return api_key


@pytest.fixture(scope="function")
def mock_llm_response():
    """
    Provide mock LLM response for testing.

    Returns:
        Mock response dictionary
    """
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mock LLM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture(scope="function")
def agent_names():
    """
    Generate agent names for multi-agent tests.

    Returns:
        List of agent names
    """
    return DataGenerator.generate_agent_names(5)


@pytest.fixture(scope="function")
def test_user_data():
    """
    Generate test user data.

    Returns:
        User data dictionary
    """
    return DataGenerator.generate_test_user()
