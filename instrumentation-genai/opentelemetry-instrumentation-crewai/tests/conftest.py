"""Pytest configuration and fixtures for CrewAI instrumentation tests."""

import os
import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


@pytest.fixture(scope="function")
def span_exporter():
    """Create an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function")
def metric_reader():
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function")
def tracer_provider(span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def meter_provider(metric_reader):
    """Create a meter provider with in-memory reader."""
    provider = MeterProvider(metric_readers=[metric_reader])
    return provider


@pytest.fixture(autouse=True)
def reset_global_handler():
    """Reset the global handler before and after each test."""
    import opentelemetry.instrumentation.crewai.instrumentation as crewai_module

    # Store original value
    original_handler = crewai_module._handler

    # Reset before test
    crewai_module._handler = None

    yield

    # Reset after test
    crewai_module._handler = original_handler


@pytest.fixture(autouse=True)
def environment():
    """Set up test environment variables."""
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters


class StubTelemetryHandler:
    """A stub telemetry handler for testing wrapper functions."""

    def __init__(self):
        self.started_workflows = []
        self.stopped_workflows = []
        self.started_agents = []
        self.stopped_agents = []
        self.started_steps = []
        self.stopped_steps = []
        self.started_tool_calls = []
        self.stopped_tool_calls = []
        self.failed_entities = []

    def start_workflow(self, workflow):
        self.started_workflows.append(workflow)
        return workflow

    def stop_workflow(self, workflow):
        self.stopped_workflows.append(workflow)
        return workflow

    def start_agent(self, agent):
        self.started_agents.append(agent)
        return agent

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)
        return agent

    def start_step(self, step):
        self.started_steps.append(step)
        return step

    def stop_step(self, step):
        self.stopped_steps.append(step)
        return step

    def start_tool_call(self, tool_call):
        self.started_tool_calls.append(tool_call)
        return tool_call

    def stop_tool_call(self, tool_call):
        self.stopped_tool_calls.append(tool_call)
        return tool_call

    def fail(self, entity, error):
        self.failed_entities.append((entity, error))
        return entity


@pytest.fixture
def stub_handler():
    """Create a stub telemetry handler for testing."""
    return StubTelemetryHandler()


# Mock CrewAI classes for testing without actual CrewAI dependency
class MockCrew:
    """Mock Crew class for testing."""

    def __init__(self, name="Test Crew"):
        self.name = name

    def kickoff(self, inputs=None):
        result = MockCrewOutput()
        result.raw = f"Completed crew '{self.name}' with inputs: {inputs}"
        return result


class MockCrewOutput:
    """Mock CrewOutput class."""

    def __init__(self):
        self.raw = ""


class MockAgent:
    """Mock Agent class for testing."""

    def __init__(self, role="Test Agent"):
        self.role = role
        self._token_process = None

    def execute_task(self, task=None, context=None, tools=None):
        return f"Agent '{self.role}' executed task"


class MockTask:
    """Mock Task class for testing."""

    def __init__(self, description="Test task", expected_output="Test output"):
        self.description = description
        self.expected_output = expected_output
        self.agent = MockAgent()

    def execute_sync(self, agent=None, context=None, tools=None):
        return f"Task '{self.description}' completed"


class MockBaseTool:
    """Mock BaseTool class for testing."""

    def __init__(self, name="test_tool"):
        self.name = name

    def run(self, *args, **kwargs):
        return f"Tool '{self.name}' executed with kwargs: {kwargs}"


class MockStructuredTool:
    """Mock CrewStructuredTool class for testing."""

    def __init__(self, name="structured_tool"):
        self.name = name

    def invoke(self, *args, **kwargs):
        return f"Structured tool '{self.name}' invoked with kwargs: {kwargs}"


@pytest.fixture
def mock_crew():
    """Create a mock Crew instance."""
    return MockCrew()


@pytest.fixture
def mock_agent():
    """Create a mock Agent instance."""
    return MockAgent()


@pytest.fixture
def mock_task():
    """Create a mock Task instance."""
    return MockTask()


@pytest.fixture
def mock_base_tool():
    """Create a mock BaseTool instance."""
    return MockBaseTool()


@pytest.fixture
def mock_structured_tool():
    """Create a mock CrewStructuredTool instance."""
    return MockStructuredTool()
