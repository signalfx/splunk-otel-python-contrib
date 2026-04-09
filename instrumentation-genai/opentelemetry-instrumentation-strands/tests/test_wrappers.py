# Copyright Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Strands wrappers."""

import pytest

from opentelemetry.instrumentation.strands.wrappers import (
    wrap_agent_call,
    wrap_agent_init,
    wrap_agent_invoke_async,
    wrap_bedrock_agentcore_app_entrypoint,
)


class MockAgent:
    """Mock Strands Agent for testing."""

    def __init__(self, name="test_agent", model="anthropic.claude-v2", tools=None):
        self.name = name
        self.model = model
        self.tools = tools or []
        self.instructions = "You are a helpful assistant"
        self.hook_registry = None

    def __call__(self, prompt):
        return f"Response to: {prompt}"


class MockAsyncAgent:
    """Mock async Strands Agent for testing."""

    def __init__(self, name="test_agent", model="anthropic.claude-v2"):
        self.name = name
        self.model = model
        self.instructions = "You are a helpful assistant"

    async def invoke_async(self, prompt):
        return f"Async response to: {prompt}"


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name):
        self.name = name


class MockHookRegistry:
    """Mock hook registry for testing."""

    def __init__(self):
        self.hooks = {}

    def register(self, event_name, callback):
        self.hooks[event_name] = callback


class MockHookProvider:
    """Mock hook provider for testing."""

    def register_hooks(self, registry):
        registry.register("test_hook", lambda: None)


def test_agent_init_injects_hook_provider(stub_handler):
    """Test that wrap_agent_init injects hook provider."""
    agent = MockAgent()
    agent.hook_registry = MockHookRegistry()
    hook_provider = MockHookProvider()

    # Create wrapper
    def original_init():
        pass

    # Wrap the init
    wrap_agent_init(original_init, agent, (), {}, hook_provider)

    # Verify hook provider was injected
    assert agent.hook_registry is not None
    assert "test_hook" in agent.hook_registry.hooks


def test_agent_call_creates_agent_invocation(stub_handler):
    """Test that wrap_agent_call creates AgentInvocation span."""
    agent = MockAgent(name="test_agent", model="anthropic.claude-v2")

    # Call wrapped method
    result = wrap_agent_call(agent.__call__, agent, ("What is AI?",), {}, stub_handler)

    # Verify invocation was started and stopped
    assert len(stub_handler.started_agents) == 1
    assert len(stub_handler.stopped_agents) == 1

    invocation = stub_handler.started_agents[0]
    assert invocation.agent_name == "test_agent"
    assert invocation.request_model == "anthropic.claude-v2"
    assert invocation.system == "strands"
    assert len(invocation.input_messages) == 1
    assert invocation.input_messages[0].parts[0].content == "What is AI?"

    # Verify result was captured
    assert result == "Response to: What is AI?"


def test_agent_call_captures_tools(stub_handler):
    """Test that wrap_agent_call captures tools list."""
    tools = [MockTool("search"), MockTool("calculator")]
    agent = MockAgent(name="test_agent", tools=tools)

    # Call wrapped method
    wrap_agent_call(agent.__call__, agent, ("Calculate 2+2",), {}, stub_handler)

    # Verify tools were captured
    invocation = stub_handler.started_agents[0]
    assert "gen_ai.tools" in invocation.attributes
    assert invocation.attributes["gen_ai.tools"] == ["search", "calculator"]


def test_agent_call_exception_fails_invocation(stub_handler):
    """Test that wrap_agent_call handles exceptions correctly."""
    agent = MockAgent()

    # Create a wrapper that raises exception
    def failing_call(*args, **kwargs):
        raise ValueError("Test error")

    # Call wrapped method and expect exception
    with pytest.raises(ValueError, match="Test error"):
        wrap_agent_call(failing_call, agent, ("prompt",), {}, stub_handler)

    # Verify invocation was started and failed
    assert len(stub_handler.started_agents) == 1
    assert len(stub_handler.failed_entities) == 1

    invocation, error = stub_handler.failed_entities[0]
    assert error.type == "ValueError"
    assert "Test error" in error.message


@pytest.mark.asyncio
async def test_async_agent_invocation(stub_handler):
    """Test that wrap_agent_invoke_async works correctly."""
    agent = MockAsyncAgent(name="async_agent", model="anthropic.claude-v2")

    # Call wrapped async method
    result = await wrap_agent_invoke_async(
        agent.invoke_async, agent, ("What is AI?",), {}, stub_handler
    )

    # Verify invocation was started and stopped
    assert len(stub_handler.started_agents) == 1
    assert len(stub_handler.stopped_agents) == 1

    invocation = stub_handler.started_agents[0]
    assert invocation.agent_name == "async_agent"
    assert invocation.request_model == "anthropic.claude-v2"

    # Verify result was captured
    assert result == "Async response to: What is AI?"


@pytest.mark.asyncio
async def test_async_agent_exception(stub_handler):
    """Test that async wrapper handles exceptions correctly."""
    agent = MockAsyncAgent()

    # Create a wrapper that raises exception
    async def failing_async_call(*args, **kwargs):
        raise RuntimeError("Async test error")

    # Call wrapped method and expect exception
    with pytest.raises(RuntimeError, match="Async test error"):
        await wrap_agent_invoke_async(
            failing_async_call, agent, ("prompt",), {}, stub_handler
        )

    # Verify invocation failed
    assert len(stub_handler.failed_entities) == 1
    invocation, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"


def test_bedrock_agentcore_app_wrapper(stub_handler):
    """Test that wrap_bedrock_agentcore_app_entrypoint creates Workflow span."""

    class MockApp:
        def __init__(self):
            self.name = "test_app"

        def entrypoint(self, event):
            return {"status": "success"}

    app = MockApp()

    # Call wrapped method
    result = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, ({"input": "test"},), {}, stub_handler
    )

    # Verify workflow was started and stopped
    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1

    workflow = stub_handler.started_workflows[0]
    assert workflow.name == "test_app"
    assert workflow.system == "strands"

    # Verify result
    assert result == {"status": "success"}


def test_bedrock_agentcore_app_exception(stub_handler):
    """Test that BedrockAgentCoreApp wrapper handles exceptions."""

    class MockApp:
        def __init__(self):
            self.name = "test_app"

        def entrypoint(self, event):
            raise ConnectionError("Service unavailable")

    app = MockApp()

    # Call wrapped method and expect exception
    with pytest.raises(ConnectionError, match="Service unavailable"):
        wrap_bedrock_agentcore_app_entrypoint(
            app.entrypoint, app, ({"input": "test"},), {}, stub_handler
        )

    # Verify workflow failed
    assert len(stub_handler.failed_entities) == 1
    workflow, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"
    assert "Service unavailable" in error.message
