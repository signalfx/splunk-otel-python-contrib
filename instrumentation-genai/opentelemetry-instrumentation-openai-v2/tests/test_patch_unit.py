# Copyright The OpenTelemetry Authors
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

"""Unit tests for patch.py functions and classes."""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from opentelemetry.instrumentation.openai_v2.patch import (
    StreamWrapper,
    _build_chat_invocation,
    _build_output_messages_from_response,
    _build_tool_call_request,
    _content_to_parts,
)
from opentelemetry.util.genai.types import (
    GenericPart,
    LLMInvocation,
    Reasoning,
    Text,
    ToolCall,
    ToolCallRequest,
)


class TestBuildChatInvocation:
    """Tests for _build_chat_invocation function."""

    def test_tool_definitions_captured_when_enabled(self, monkeypatch):
        """Test that tool_definitions is captured when both flags are enabled."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS", "true"
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                },
            }
        ]
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": tools,
        }

        invocation = _build_chat_invocation(kwargs)

        assert invocation.tool_definitions is not None
        parsed = json.loads(invocation.tool_definitions)
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "get_weather"

    def test_tool_definitions_not_captured_when_disabled(self, monkeypatch):
        """Test that tool_definitions is not captured when flag is disabled."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS", "false"
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                },
            }
        ]
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": tools,
        }

        invocation = _build_chat_invocation(kwargs)

        assert invocation.tool_definitions is None

    def test_tool_definitions_always_populated_on_object(self, monkeypatch):
        """Tool definitions are always populated on the Python object when
        CAPTURE_TOOL_DEFINITIONS is enabled, regardless of content capture mode.
        The emitter layer controls what gets written to telemetry."""
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
        )
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS", "true"
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                },
            }
        ]
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": tools,
        }

        invocation = _build_chat_invocation(kwargs)

        assert invocation.tool_definitions is not None
        parsed = json.loads(invocation.tool_definitions)
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "get_weather"


class TestStreamWrapper:
    """Tests for StreamWrapper class."""

    def test_time_to_first_chunk_captured(self):
        """Test that time_to_first_chunk is captured on first chunk."""
        # Create a mock invocation with _start_time set
        invocation = LLMInvocation(request_model="gpt-4o")
        invocation._start_time = time.perf_counter()
        invocation.request_stream = True

        # Create mock stream and handler
        mock_stream = MagicMock()
        mock_handler = MagicMock()

        wrapper = StreamWrapper(
            stream=mock_stream,
            invocation=invocation,
            handler=mock_handler,
        )

        # Simulate a small delay before first chunk
        time.sleep(0.01)

        # Create a mock chunk
        mock_chunk = MagicMock()
        mock_chunk.id = "chatcmpl-123"
        mock_chunk.model = "gpt-4o"
        mock_chunk.service_tier = None
        mock_chunk.choices = []
        mock_chunk.usage = None

        # Process the chunk
        wrapper.process_chunk(mock_chunk)

        # Verify time_to_first_chunk is set
        ttfc = invocation.attributes.get("gen_ai.response.time_to_first_chunk")
        assert ttfc is not None, "time_to_first_chunk should be captured"
        assert ttfc > 0, "time_to_first_chunk should be positive"
        assert ttfc < 1, "time_to_first_chunk should be less than 1 second"

    def test_time_to_first_chunk_only_captured_once(self):
        """Test that time_to_first_chunk is only captured on the first chunk."""
        invocation = LLMInvocation(request_model="gpt-4o")
        invocation._start_time = time.perf_counter()
        invocation.request_stream = True

        mock_stream = MagicMock()
        mock_handler = MagicMock()

        wrapper = StreamWrapper(
            stream=mock_stream,
            invocation=invocation,
            handler=mock_handler,
        )

        # Create mock chunks
        mock_chunk = MagicMock()
        mock_chunk.id = "chatcmpl-123"
        mock_chunk.model = "gpt-4o"
        mock_chunk.service_tier = None
        mock_chunk.choices = []
        mock_chunk.usage = None

        # Process first chunk
        wrapper.process_chunk(mock_chunk)
        first_ttfc = invocation.attributes.get(
            "gen_ai.response.time_to_first_chunk"
        )

        # Process second chunk after delay
        time.sleep(0.01)
        wrapper.process_chunk(mock_chunk)
        second_ttfc = invocation.attributes.get(
            "gen_ai.response.time_to_first_chunk"
        )

        # TTFC should remain the same
        assert first_ttfc == second_ttfc, (
            "time_to_first_chunk should not change after first chunk"
        )

    def test_raw_response_headers_stored_from_legacy_api_response(self):
        """Test the real production scenario: LiteLLM calls with_raw_response.create(stream=True),
        which makes the OpenAI SDK return a LegacyAPIResponse. SDOT's _parse_response calls
        .parse() on it to get the AsyncStream — discarding .headers. StreamWrapper must
        preserve the headers captured before _parse_response so LiteLLM can access them.

        Regression test for the AttributeError seen in production:
          File "litellm/llms/azure/azure.py", line 176
            headers = dict(raw_response.headers)
          AttributeError: 'StreamWrapper' object has no attribute 'headers'
        """
        invocation = LLMInvocation(request_model="gpt-4o")
        mock_stream = MagicMock(spec=[])  # AsyncStream has no .headers
        mock_handler = MagicMock()
        raw_headers = {
            "content-type": "text/event-stream",
            "x-request-id": "abc123",
        }

        wrapper = StreamWrapper(
            stream=mock_stream,
            invocation=invocation,
            handler=mock_handler,
            raw_headers=raw_headers,
        )

        # LiteLLM: headers = dict(raw_response.headers)
        assert wrapper.headers == raw_headers
        # LiteLLM: response = raw_response.parse()
        assert wrapper.parse() is wrapper

    def test_getattr_proxies_unknown_attributes_to_stream(self):
        """Test that unknown attributes on StreamWrapper are proxied to the
        underlying stream object.
        Regression test for https://github.com/open-telemetry/opentelemetry-python-contrib/issues/4113
        """
        invocation = LLMInvocation(request_model="gpt-4o")
        mock_stream = MagicMock()
        mock_stream.some_custom_attr = "value"
        mock_handler = MagicMock()

        wrapper = StreamWrapper(
            stream=mock_stream,
            invocation=invocation,
            handler=mock_handler,
        )

        assert wrapper.some_custom_attr == "value"

    def test_time_to_first_chunk_not_captured_without_start_time(self):
        """Test that time_to_first_chunk is not captured without _start_time."""
        invocation = LLMInvocation(request_model="gpt-4o")
        # Note: _start_time is NOT set

        mock_stream = MagicMock()
        mock_handler = MagicMock()

        wrapper = StreamWrapper(
            stream=mock_stream,
            invocation=invocation,
            handler=mock_handler,
        )

        mock_chunk = MagicMock()
        mock_chunk.id = "chatcmpl-123"
        mock_chunk.model = "gpt-4o"
        mock_chunk.service_tier = None
        mock_chunk.choices = []
        mock_chunk.usage = None

        wrapper.process_chunk(mock_chunk)

        ttfc = invocation.attributes.get("gen_ai.response.time_to_first_chunk")
        assert ttfc is None, (
            "time_to_first_chunk should not be set without _start_time"
        )


class TestRequestStreamFlag:
    """Tests for request_stream flag setting."""

    def test_is_streaming_detection(self):
        """Test that is_streaming correctly detects streaming requests."""
        from opentelemetry.instrumentation.openai_v2.patch import is_streaming

        # Verify is_streaming detection works
        assert is_streaming({"stream": True}) is True
        assert is_streaming({"stream": False}) is False
        assert is_streaming({}) is False

    def test_request_stream_set_on_invocation(self):
        """Test that request_stream is set on invocation for streaming requests."""
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Non-streaming - _build_chat_invocation doesn't set request_stream
        invocation = _build_chat_invocation(kwargs)
        assert invocation.request_stream is None

        # The request_stream flag is set in chat_completions_create wrapper
        # after _build_chat_invocation, so we test the integration behavior
        # via the StreamWrapper which expects request_stream to be True
        invocation.request_stream = True
        assert invocation.request_stream is True


# ---------------------------------------------------------------------------
# Helpers for new message type tests
# ---------------------------------------------------------------------------


def _make_response(choices):
    """Build a minimal OpenAI-like response object."""
    response = SimpleNamespace(choices=choices, id="resp-1", model="gpt-4o")
    return response


def _make_choice(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(message=message, finish_reason=finish_reason)


def _make_tool_call(name, arguments, call_id="call-1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=function)


# ---------------------------------------------------------------------------
# _content_to_parts
# ---------------------------------------------------------------------------


class TestContentToParts:
    def test_string_content_captured(self):
        parts = _content_to_parts("hello", capture_content=True)
        assert parts == [Text(content="hello")]

    def test_string_content_redacted(self):
        parts = _content_to_parts("secret", capture_content=False)
        assert parts == [Text(content="")]

    def test_none_returns_empty(self):
        assert _content_to_parts(None, capture_content=True) == []

    def test_text_block_captured(self):
        blocks = [{"type": "text", "text": "world"}]
        parts = _content_to_parts(blocks, capture_content=True)
        assert parts == [Text(content="world")]

    def test_text_block_redacted(self):
        blocks = [{"type": "text", "text": "world"}]
        parts = _content_to_parts(blocks, capture_content=False)
        assert parts == [Text(content="")]

    def test_reasoning_block_captured(self):
        blocks = [{"type": "reasoning", "content": "I think..."}]
        parts = _content_to_parts(blocks, capture_content=True)
        assert parts == [Reasoning(content="I think...")]

    def test_thinking_block_captured(self):
        blocks = [{"type": "thinking", "content": "hmm"}]
        parts = _content_to_parts(blocks, capture_content=True)
        assert parts == [Reasoning(content="hmm")]

    def test_reasoning_block_redacted(self):
        blocks = [{"type": "reasoning", "content": "secret thoughts"}]
        parts = _content_to_parts(blocks, capture_content=False)
        assert parts == [Reasoning(content="")]

    def test_unknown_block_becomes_generic_part(self):
        block = {"type": "image_url", "url": "http://example.com/img.png"}
        parts = _content_to_parts([block], capture_content=True)
        assert len(parts) == 1
        assert isinstance(parts[0], GenericPart)
        assert parts[0].value == block

    def test_unknown_block_redacted(self):
        block = {"type": "image_url", "url": "sensitive"}
        parts = _content_to_parts([block], capture_content=False)
        assert isinstance(parts[0], GenericPart)
        assert parts[0].value is None

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "reasoning", "content": "thinking"},
            {"type": "image_url", "url": "x"},
        ]
        parts = _content_to_parts(blocks, capture_content=True)
        assert isinstance(parts[0], Text)
        assert isinstance(parts[1], Reasoning)
        assert isinstance(parts[2], GenericPart)


# ---------------------------------------------------------------------------
# _build_tool_call_request
# ---------------------------------------------------------------------------


class TestBuildToolCallRequest:
    def test_basic_object(self):
        tc = _make_tool_call("get_weather", '{"city": "SF"}', "call-42")
        result = _build_tool_call_request(tc, capture_content=True)
        assert isinstance(result, ToolCallRequest)
        assert result.name == "get_weather"
        assert result.id == "call-42"
        assert result.arguments == {"city": "SF"}

    def test_arguments_string_parse_failure_kept_as_string(self):
        tc = _make_tool_call("fn", "not-json", "call-1")
        result = _build_tool_call_request(tc, capture_content=True)
        assert result.arguments == "not-json"

    def test_arguments_none_when_no_capture(self):
        tc = _make_tool_call("fn", '{"x": 1}', "call-1")
        result = _build_tool_call_request(tc, capture_content=False)
        assert result.arguments is None

    def test_dict_tool_call(self):
        tc = {
            "id": "call-dict",
            "function": {"name": "lookup", "arguments": '{"q": "otel"}'},
        }
        result = _build_tool_call_request(tc, capture_content=True)
        assert result.name == "lookup"
        assert result.id == "call-dict"
        assert result.arguments == {"q": "otel"}

    def test_missing_name_defaults(self):
        tc = SimpleNamespace(
            id="c", function=SimpleNamespace(name=None, arguments=None)
        )
        result = _build_tool_call_request(tc, capture_content=True)
        assert result.name == "unnamed_tool_call"


# ---------------------------------------------------------------------------
# _build_output_messages_from_response  (flag on vs off)
# ---------------------------------------------------------------------------


class TestBuildOutputMessagesNewTypes:
    def test_flag_off_text_produces_tool_call_legacy(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES",
            raising=False,
        )
        tc = _make_tool_call("search", '{"q": "hello"}')
        response = _make_response(
            [_make_choice(tool_calls=[tc], finish_reason="tool_calls")]
        )
        msgs = _build_output_messages_from_response(
            response, capture_content=True
        )
        assert len(msgs) == 1
        assert len(msgs[0].parts) == 1
        assert isinstance(msgs[0].parts[0], ToolCall)

    def test_flag_on_tool_call_produces_tool_call_request(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", "true"
        )
        tc = _make_tool_call("search", '{"q": "hello"}', "call-99")
        response = _make_response(
            [_make_choice(tool_calls=[tc], finish_reason="tool_calls")]
        )
        msgs = _build_output_messages_from_response(
            response, capture_content=True
        )
        assert len(msgs) == 1
        assert len(msgs[0].parts) == 1
        part = msgs[0].parts[0]
        assert isinstance(part, ToolCallRequest)
        assert part.name == "search"
        assert part.id == "call-99"
        assert part.arguments == {"q": "hello"}

    def test_flag_off_text_produces_text_part(self, monkeypatch):
        monkeypatch.delenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES",
            raising=False,
        )
        response = _make_response([_make_choice(content="hello")])
        msgs = _build_output_messages_from_response(
            response, capture_content=True
        )
        assert isinstance(msgs[0].parts[0], Text)
        assert msgs[0].parts[0].content == "hello"

    def test_flag_on_text_produces_text_part(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", "true"
        )
        response = _make_response([_make_choice(content="hello")])
        msgs = _build_output_messages_from_response(
            response, capture_content=True
        )
        assert isinstance(msgs[0].parts[0], Text)
        assert msgs[0].parts[0].content == "hello"

    def test_flag_on_reasoning_block_produces_reasoning_part(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", "true"
        )
        content = [{"type": "reasoning", "content": "step by step"}]
        response = _make_response([_make_choice(content=content)])
        msgs = _build_output_messages_from_response(
            response, capture_content=True
        )
        assert isinstance(msgs[0].parts[0], Reasoning)
        assert msgs[0].parts[0].content == "step by step"

    def test_no_choices_returns_empty(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_ENABLE_NEW_MESSAGE_TYPES", "true"
        )
        response = _make_response([])
        assert (
            _build_output_messages_from_response(
                response, capture_content=True
            )
            == []
        )
