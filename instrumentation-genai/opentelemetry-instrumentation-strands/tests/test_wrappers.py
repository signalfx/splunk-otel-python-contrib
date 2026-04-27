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
    wrap_agent_init,
    wrap_agent_invoke_async,
    wrap_bedrock_agentcore_app_entrypoint,
    wrap_stream_messages,
)
from opentelemetry.util.genai.types import LLMInvocation


class MockModel:
    """Mock BedrockModel with config dict matching real BedrockModel layout."""

    def __init__(self, model_id="anthropic.claude-v2"):
        self.config = {"model_id": model_id}


class MockAgent:
    """Mock Strands Agent for testing."""

    def __init__(self, name="test_agent", model_id="anthropic.claude-v2", tools=None):
        self.name = name
        self.model = MockModel(model_id)
        self.tools = tools or []
        self.system_prompt = "You are a helpful assistant"
        # Real Agent uses `hooks` not `hook_registry`
        self.hooks = None

    async def invoke_async(self, prompt):
        return f"Async response to: {prompt}"


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name):
        self.name = name


class MockHookRegistry:
    """Mock hook registry using the real add_callback API."""

    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event_type, callback):
        self.callbacks[event_type] = callback


class MockHookProvider:
    """Mock hook provider for testing."""

    def register_hooks(self, registry):
        registry.add_callback("test_event", lambda: None)


def test_agent_init_injects_hook_provider(stub_handler):
    """wrap_agent_init should inject the hook provider into agent.hooks."""
    agent = MockAgent()
    agent.hooks = MockHookRegistry()
    hook_provider = MockHookProvider()

    wrap_agent_init(lambda: None, agent, (), {}, hook_provider)

    assert "test_event" in agent.hooks.callbacks


def test_agent_init_missing_hooks_attr(stub_handler):
    """wrap_agent_init should not crash if agent has no hooks attribute."""
    agent = MockAgent()
    agent.hooks = None
    hook_provider = MockHookProvider()

    # Should not raise
    wrap_agent_init(lambda: None, agent, (), {}, hook_provider)


@pytest.mark.asyncio
async def test_async_agent_invocation_with_conversation_id(stub_handler):
    """wrap_agent_invoke_async should extract conversation_id from invocation_state."""
    agent = MockAgent(name="async_agent", model_id="anthropic.claude-v2")

    # Mock invoke_async that accepts invocation_state
    async def invoke_with_state(prompt, invocation_state=None):
        return f"Response to: {prompt}"

    result = await wrap_agent_invoke_async(
        invoke_with_state,
        agent,
        ("What is AI?",),
        {"invocation_state": {"conversation_id": "test-conversation-123"}},
        stub_handler,
    )

    assert len(stub_handler.started_agents) == 1
    invocation = stub_handler.started_agents[0]
    assert invocation.conversation_id == "test-conversation-123"
    assert result == "Response to: What is AI?"


@pytest.mark.asyncio
async def test_async_agent_invocation(stub_handler):
    """wrap_agent_invoke_async should create an AgentInvocation span."""
    agent = MockAgent(name="async_agent", model_id="anthropic.claude-v2")

    result = await wrap_agent_invoke_async(
        agent.invoke_async, agent, ("What is AI?",), {}, stub_handler
    )

    assert len(stub_handler.started_agents) == 1
    assert len(stub_handler.stopped_agents) == 1

    invocation = stub_handler.started_agents[0]
    assert invocation.name == "async_agent"
    assert invocation.model == "anthropic.claude-v2"
    assert invocation.system == "strands"
    assert len(invocation.input_messages) == 1
    assert invocation.input_messages[0].parts[0].content == "What is AI?"

    assert result == "Async response to: What is AI?"


@pytest.mark.asyncio
async def test_async_agent_exception(stub_handler):
    """wrap_agent_invoke_async should fail the invocation on exception."""
    agent = MockAgent()

    async def failing_async_call(*args, **kwargs):
        raise RuntimeError("Async test error")

    with pytest.raises(RuntimeError, match="Async test error"):
        await wrap_agent_invoke_async(
            failing_async_call, agent, ("prompt",), {}, stub_handler
        )

    assert len(stub_handler.failed_entities) == 1
    _invocation, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"


def test_bedrock_agentcore_app_wrapper_sync(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should create a Workflow span for sync functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func  # simple passthrough decorator

    app = MockApp()

    def my_handler(payload):
        return {"status": "success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (my_handler,), {}, stub_handler
    )
    result = wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    workflow = stub_handler.started_workflows[0]
    assert workflow.name == "test_app"
    assert workflow.system == "strands"
    assert result == {"status": "success"}


def test_bedrock_agentcore_app_wrapper_sync_exception(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should fail the Workflow on exception."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()

    def failing_handler(payload):
        raise ConnectionError("Service unavailable")

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (failing_handler,), {}, stub_handler
    )

    with pytest.raises(ConnectionError, match="Service unavailable"):
        wrapped({})

    assert len(stub_handler.failed_entities) == 1
    _workflow, error = stub_handler.failed_entities[0]
    assert error.type == "ConnectionError"


@pytest.mark.asyncio
async def test_wrap_stream_messages_sets_token_counts():
    """wrap_stream_messages should populate input/output tokens on the active LLMInvocation."""
    from strands.event_loop.streaming import ModelStopReason
    from strands.types.event_loop import Metrics, Usage

    invocation_state = {}
    invocation = LLMInvocation(request_model="anthropic.claude-v2")

    class MockHookProvider:
        _active_llm_invocations = {id(invocation_state): invocation}

    usage = Usage(inputTokens=42, outputTokens=17, totalTokens=59)
    metrics = Metrics(latencyMs=100)
    stop_event = ModelStopReason(
        stop_reason="end_turn",
        message={"role": "assistant", "content": []},
        usage=usage,
        metrics=metrics,
    )

    async def fake_stream_messages(*args, **kwargs):
        yield {"contentBlockDelta": {}}
        yield stop_event

    events = []
    async for event in wrap_stream_messages(
        fake_stream_messages,
        None,
        (),
        {"invocation_state": invocation_state},
        MockHookProvider(),
    ):
        events.append(event)

    assert len(events) == 2
    assert invocation.input_tokens == 42
    assert invocation.output_tokens == 17


@pytest.mark.asyncio
async def test_wrap_stream_messages_no_invocation_state():
    """wrap_stream_messages should pass through events when invocation_state is absent."""

    async def fake_stream(*args, **kwargs):
        yield {"contentBlockDelta": {}}

    events = []
    async for event in wrap_stream_messages(
        fake_stream, None, (), {}, object()
    ):
        events.append(event)

    assert len(events) == 1


@pytest.mark.asyncio
async def test_bedrock_agentcore_app_wrapper_async(stub_handler):
    """wrap_bedrock_agentcore_app_entrypoint should handle async entrypoint functions."""

    class MockApp:
        name = "test_app"

        def entrypoint(self, func):
            return func

    app = MockApp()

    async def async_handler(payload):
        return {"status": "async_success"}

    wrapped = wrap_bedrock_agentcore_app_entrypoint(
        app.entrypoint, app, (async_handler,), {}, stub_handler
    )
    result = await wrapped({"input": "test"})

    assert len(stub_handler.started_workflows) == 1
    assert len(stub_handler.stopped_workflows) == 1
    assert result == {"status": "async_success"}
