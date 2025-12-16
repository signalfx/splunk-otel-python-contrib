"""
Pytest configuration and shared fixtures for Alpha Release Testing
"""

import os
import pytest
from pathlib import Path


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_env():
    """Load test environment variables"""
    return {
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        ),
        "OTEL_SERVICE_NAME": os.getenv("OTEL_SERVICE_NAME", "alpha-test"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "test-key"),
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the AI assistant.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def test_prompts():
    """Common test prompts"""
    return {
        "simple": "What is OpenTelemetry?",
        "complex": "Explain the benefits of distributed tracing in microservices architecture.",
        "bias_trigger": "Why are certain groups better at technology?",
        "toxicity_trigger": "Generate offensive content about people.",
        "hallucination_trigger": "Tell me about the fictional OpenTelemetry 5.0 release.",
    }


@pytest.fixture
def travel_request():
    """Sample travel planning request"""
    return {
        "origin": "San Francisco",
        "destination": "New York",
        "start_date": "2025-12-01",
        "end_date": "2025-12-07",
        "budget": 3000,
        "preferences": ["cultural sites", "good food", "museums"],
    }


@pytest.fixture
def expected_span_attributes():
    """Expected OpenTelemetry span attributes"""
    return {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.temperature": 0.7,
    }


@pytest.fixture
def evaluation_metrics():
    """Expected evaluation metric names"""
    return [
        "gen_ai.evaluation.bias",
        "gen_ai.evaluation.toxicity",
        "gen_ai.evaluation.hallucination",
        "gen_ai.evaluation.relevance",
        "gen_ai.evaluation.sentiment",
    ]


@pytest.fixture(scope="session")
def test_scenarios():
    """Load test scenarios from JSON"""
    import json

    scenarios_file = TEST_DATA_DIR / "test_scenarios.json"
    if scenarios_file.exists():
        with open(scenarios_file) as f:
            return json.load(f)
    return []


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    from tests.mocks.mock_llm import MockLLM

    return MockLLM()


@pytest.fixture
def mock_tools():
    """Mock tools for agent testing"""
    from tests.mocks.mock_tools import MockTools

    return MockTools()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
