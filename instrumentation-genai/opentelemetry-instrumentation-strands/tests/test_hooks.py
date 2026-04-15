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

from unittest import mock

from opentelemetry.instrumentation.strands.hooks import StrandsHookProvider


class MockModel:
    """Mock BedrockModel with config dict (matches actual BedrockModel.config)."""

    def __init__(self, model_id="anthropic.claude-v2"):
        self.config = {"model_id": model_id}


class MockAgent:
    """Mock Strands Agent attached to hook events."""

    def __init__(self, model_id="anthropic.claude-v2"):
        self.model = MockModel(model_id)


class MockStopResponse:
    """Mock AfterModelCallEvent.stop_response (ModelStopResponse dataclass)."""

    def __init__(self, text="response text", stop_reason="end_turn"):
        # message is a Message TypedDict: {"role": ..., "content": [...]}
        self.message = {
            "role": "assistant",
            "content": [{"text": text}],
        }
        self.stop_reason = stop_reason


def _make_before_model_event(model_id="anthropic.claude-v2", messages=None):
    """Build a mock BeforeModelCallEvent with the real field layout."""
    agent = MockAgent(model_id)
    invocation_state = {"messages": messages or [{"role": "user", "content": "Hello"}]}
    event = mock.MagicMock()
    event.agent = agent
    event.invocation_state = invocation_state
    return event, invocation_state


def _make_after_model_event(invocation_state, stop_response=None, exception=None):
    """Build a mock AfterModelCallEvent."""
    event = mock.MagicMock()
    event.invocation_state = invocation_state
    event.stop_response = stop_response
    event.exception = exception
    return event


def _make_before_tool_event(
    name="search_docs", tool_use_id="tool-123", input_data=None
):
    """Build a mock BeforeToolCallEvent with tool_use TypedDict layout."""
    event = mock.MagicMock()
    event.tool_use = {
        "name": name,
        "toolUseId": tool_use_id,
        "input": input_data or {"query": "AI trends"},
    }
    return event


def _make_after_tool_event(tool_use_id="tool-123", result=None, exception=None):
    """Build a mock AfterToolCallEvent."""
    event = mock.MagicMock()
    event.tool_use = {"name": "search_docs", "toolUseId": tool_use_id, "input": {}}
    event.result = result
    event.exception = exception
    return event


def test_before_model_call_starts_llm(stub_handler):
    """BeforeModelCallEvent should start an LLM invocation with correct model and messages."""
    hook_provider = StrandsHookProvider(stub_handler)

    event, _ = _make_before_model_event(
        model_id="anthropic.claude-v2",
        messages=[{"role": "user", "content": "Hello"}],
    )
    hook_provider._on_before_model_call(event)

    assert len(stub_handler.started_llm) == 1
    invocation = stub_handler.started_llm[0]
    assert invocation.request_model == "anthropic.claude-v2"
    assert invocation.system == "aws.bedrock"
    assert len(invocation.input_messages) == 1
    assert invocation.input_messages[0].role == "user"


def test_after_model_call_stops_llm(stub_handler):
    """AfterModelCallEvent should stop the matching LLM invocation and populate response."""
    hook_provider = StrandsHookProvider(stub_handler)

    event, invocation_state = _make_before_model_event()
    hook_provider._on_before_model_call(event)

    stop_response = MockStopResponse(
        text="Hello! How can I help?", stop_reason="end_turn"
    )
    after_event = _make_after_model_event(invocation_state, stop_response=stop_response)
    hook_provider._on_after_model_call(after_event)

    assert len(stub_handler.stopped_llm) == 1
    invocation = stub_handler.stopped_llm[0]
    assert len(invocation.output_messages) == 1
    assert "Hello! How can I help?" in invocation.output_messages[0].parts[0].content
    assert invocation.output_messages[0].finish_reason == "end_turn"


def test_after_model_call_with_exception_fails_llm(stub_handler):
    """AfterModelCallEvent with exception should fail the LLM invocation."""
    hook_provider = StrandsHookProvider(stub_handler)

    event, invocation_state = _make_before_model_event()
    hook_provider._on_before_model_call(event)

    after_event = _make_after_model_event(
        invocation_state, exception=ValueError("API error")
    )
    hook_provider._on_after_model_call(after_event)

    assert len(stub_handler.failed_entities) == 1
    _invocation, error = stub_handler.failed_entities[0]
    assert error.type == "ValueError"
    assert "API error" in error.message


def test_before_tool_call_starts_tool_call(stub_handler):
    """BeforeToolCallEvent should start a tool call with name and arguments from tool_use."""
    hook_provider = StrandsHookProvider(stub_handler)

    event = _make_before_tool_event(
        name="search_docs", tool_use_id="tool-123", input_data={"query": "AI trends"}
    )
    hook_provider._on_before_tool_call(event)

    assert len(stub_handler.started_tool_calls) == 1
    tool_call = stub_handler.started_tool_calls[0]
    assert tool_call.name == "search_docs"
    assert tool_call.id == "tool-123"
    assert tool_call.system == "strands"
    assert '"query": "AI trends"' in tool_call.arguments


def test_after_tool_call_stops_tool_call(stub_handler):
    """AfterToolCallEvent should stop the matching tool call and capture result."""
    hook_provider = StrandsHookProvider(stub_handler)

    hook_provider._on_before_tool_call(_make_before_tool_event(tool_use_id="tool-123"))
    hook_provider._on_after_tool_call(
        _make_after_tool_event(
            tool_use_id="tool-123", result={"documents": ["doc1", "doc2"]}
        )
    )

    assert len(stub_handler.stopped_tool_calls) == 1
    tool_call = stub_handler.stopped_tool_calls[0]
    assert '"documents"' in tool_call.tool_result


def test_after_tool_call_with_exception_fails_tool_call(stub_handler):
    """AfterToolCallEvent with exception should fail the tool call."""
    hook_provider = StrandsHookProvider(stub_handler)

    hook_provider._on_before_tool_call(_make_before_tool_event(tool_use_id="tool-123"))
    hook_provider._on_after_tool_call(
        _make_after_tool_event(
            tool_use_id="tool-123", exception=RuntimeError("Tool execution failed")
        )
    )

    assert len(stub_handler.failed_entities) == 1
    _tool_call, error = stub_handler.failed_entities[0]
    assert error.type == "RuntimeError"
    assert "Tool execution failed" in error.message


def test_extract_provider_from_model_id(stub_handler):
    """_extract_provider should return the correct provider for known model ID patterns."""
    hook_provider = StrandsHookProvider(stub_handler)

    assert hook_provider._extract_provider("anthropic.claude-v2") == "anthropic"
    assert hook_provider._extract_provider("openai.gpt-4") == "openai"
    assert hook_provider._extract_provider("cohere.command") == "cohere"
    assert hook_provider._extract_provider("ai21.j2-ultra") == "ai21"
    assert hook_provider._extract_provider("") == "unknown"


def test_hook_provider_register_hooks(stub_handler):
    """register_hooks should call add_callback with event type classes."""
    from strands.hooks.events import (
        AfterModelCallEvent,
        AfterToolCallEvent,
        BeforeModelCallEvent,
        BeforeToolCallEvent,
    )

    hook_provider = StrandsHookProvider(stub_handler)

    class CapturingRegistry:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, event_type, callback):
            self.callbacks[event_type] = callback

    registry = CapturingRegistry()
    hook_provider.register_hooks(registry)

    assert BeforeModelCallEvent in registry.callbacks
    assert AfterModelCallEvent in registry.callbacks
    assert BeforeToolCallEvent in registry.callbacks
    assert AfterToolCallEvent in registry.callbacks
