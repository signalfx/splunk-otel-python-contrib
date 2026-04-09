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

"""Tests for StrandsHookProvider."""

from opentelemetry.instrumentation.strands.hooks import StrandsHookProvider


class MockEvent:
    """Mock event for testing hook callbacks."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockInvocationState:
    """Mock invocation state for tracking LLM invocations."""

    pass


class MockUsage:
    """Mock usage object."""

    def __init__(self, prompt_tokens=10, completion_tokens=20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockResponse:
    """Mock LLM response object."""

    def __init__(self, content="response", model="anthropic.claude-v2", usage=None):
        self.content = content
        self.model = model
        self.usage = usage or MockUsage()
        self.finish_reason = "stop"


def test_before_model_call_starts_llm(stub_handler):
    """Test that BeforeModelCallEvent starts LLM invocation."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create mock event
    invocation_state = MockInvocationState()
    event = MockEvent(
        model="anthropic.claude-v2",
        messages=[{"role": "user", "content": "Hello"}],
        invocation_state=invocation_state,
        temperature=0.7,
        max_tokens=100,
    )

    # Call handler
    hook_provider._on_before_model_call(event)

    # Verify LLM invocation started
    assert len(stub_handler.started_llm) == 1
    invocation = stub_handler.started_llm[0]
    assert invocation.request_model == "anthropic.claude-v2"
    assert invocation.system == "strands"
    assert len(invocation.input_messages) == 1
    assert invocation.input_messages[0].role == "user"
    assert invocation.request_temperature == 0.7
    assert invocation.request_max_tokens == 100


def test_after_model_call_stops_llm(stub_handler):
    """Test that AfterModelCallEvent stops LLM invocation."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create and start invocation
    invocation_state = MockInvocationState()
    before_event = MockEvent(
        model="anthropic.claude-v2",
        messages=[{"role": "user", "content": "Hello"}],
        invocation_state=invocation_state,
    )
    hook_provider._on_before_model_call(before_event)

    # Create after event with response
    response = MockResponse(
        content="Hello! How can I help?",
        model="anthropic.claude-v2",
        usage=MockUsage(prompt_tokens=5, completion_tokens=10),
    )
    after_event = MockEvent(invocation_state=invocation_state, response=response)

    # Call handler
    hook_provider._on_after_model_call(after_event)

    # Verify LLM invocation stopped
    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]
    assert invocation.response_model == "anthropic.claude-v2"
    assert invocation.usage_input_tokens == 5
    assert invocation.usage_output_tokens == 10
    assert len(invocation.output_messages) == 1
    assert invocation.output_messages[0].parts[0].content == "Hello! How can I help?"


def test_after_model_call_with_exception_fails_llm(stub_handler):
    """Test that AfterModelCallEvent with exception fails LLM invocation."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create and start invocation
    invocation_state = MockInvocationState()
    before_event = MockEvent(
        model="anthropic.claude-v2",
        messages=[{"role": "user", "content": "Hello"}],
        invocation_state=invocation_state,
    )
    hook_provider._on_before_model_call(before_event)

    # Create after event with exception
    exception = ValueError("API error")
    after_event = MockEvent(invocation_state=invocation_state, exception=exception)

    # Call handler
    hook_provider._on_after_model_call(after_event)

    # Verify LLM invocation failed
    assert len(stub_handler.failed_entities) == 1
    invocation, error = stub_handler.failed_entities[0]
    assert error.type == "ValueError"
    assert "API error" in error.message


def test_before_tool_call_starts_tool_call(stub_handler):
    """Test that BeforeToolCallEvent starts tool call."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create mock event
    event = MockEvent(
        tool_name="search_docs",
        tool_use_id="tool-123",
        arguments={"query": "AI trends"},
        description="Search documentation",
    )

    # Call handler
    hook_provider._on_before_tool_call(event)

    # Verify tool call started
    assert len(stub_handler.started_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "search_docs"
    assert tool_call.id == "tool-123"
    assert tool_call.system == "strands"
    assert '"query": "AI trends"' in tool_call.arguments


def test_after_tool_call_stops_tool_call(stub_handler):
    """Test that AfterToolCallEvent stops tool call."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create and start tool call
    before_event = MockEvent(
        tool_name="search_docs",
        tool_use_id="tool-123",
        arguments={"query": "AI trends"},
    )
    hook_provider._on_before_tool_call(before_event)

    # Create after event with result
    after_event = MockEvent(
        tool_use_id="tool-123", result={"documents": ["doc1", "doc2"]}
    )

    # Call handler
    hook_provider._on_after_tool_call(after_event)

    # Verify tool call stopped
    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.stopped_tool_calls[0]
    assert '"documents"' in tool_call.tool_result


def test_after_tool_call_with_exception_fails_tool_call(stub_handler):
    """Test that AfterToolCallEvent with exception fails tool call."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create and start tool call
    before_event = MockEvent(
        tool_name="search_docs",
        tool_use_id="tool-123",
        arguments={"query": "AI trends"},
    )
    hook_provider._on_before_tool_call(before_event)

    # Create after event with exception
    exception = RuntimeError("Tool execution failed")
    after_event = MockEvent(tool_use_id="tool-123", exception=exception)

    # Call handler
    hook_provider._on_after_tool_call(after_event)

    # Verify tool call failed
    assert len(stub_handler.failed_entities) == 1
    tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"
    assert "Tool execution failed" in error.message


def test_extract_provider_from_model_id(stub_handler):
    """Test provider extraction from various model ID formats."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Test various model ID formats
    assert hook_provider._extract_provider("anthropic.claude-v2") == "anthropic"
    assert hook_provider._extract_provider("bedrock-anthropic.claude-v2") == "bedrock"
    assert hook_provider._extract_provider("openai.gpt-4") == "openai"
    assert hook_provider._extract_provider("cohere.command") == "cohere"
    assert hook_provider._extract_provider("ai21.j2-ultra") == "ai21"
    assert hook_provider._extract_provider("unknown-model") == "unknown-model"
    assert hook_provider._extract_provider("") == "unknown"


def test_hook_provider_register_hooks(stub_handler):
    """Test that register_hooks is callable and doesn't crash."""
    hook_provider = StrandsHookProvider(stub_handler)

    # Create a mock registry
    class MockRegistry:
        def __init__(self):
            self.hooks = {}

        def register(self, event_name, callback):
            self.hooks[event_name] = callback

    registry = MockRegistry()

    # Register hooks
    hook_provider.register_hooks(registry)

    # Verify hooks were registered
    assert "BeforeModelCallEvent" in registry.hooks
    assert "AfterModelCallEvent" in registry.hooks
    assert "BeforeToolCallEvent" in registry.hooks
    assert "AfterToolCallEvent" in registry.hooks
