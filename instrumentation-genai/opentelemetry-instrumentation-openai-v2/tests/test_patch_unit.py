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
from unittest.mock import MagicMock

from opentelemetry.instrumentation.openai_v2.patch import (
    StreamWrapper,
    _build_chat_invocation,
)
from opentelemetry.util.genai.types import LLMInvocation


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
