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

"""Tests for FastMCP instrumentation utility functions."""

import os
from unittest.mock import patch

from opentelemetry.instrumentation.fastmcp.utils import (
    safe_serialize,
    truncate_if_needed,
    should_capture_content,
    is_instrumentation_enabled,
    extract_tool_info,
    extract_result_content,
    detect_client_network_transport,
    detect_server_network_transport,
)


class TestSafeSerialize:
    """Tests for safe_serialize function."""

    def test_none_input(self):
        """Test serializing None returns None."""
        assert safe_serialize(None) is None

    def test_simple_dict(self):
        """Test serializing a simple dictionary."""
        result = safe_serialize({"key": "value", "num": 42})
        assert result is not None
        assert '"key"' in result
        assert '"value"' in result

    def test_nested_dict(self):
        """Test serializing nested dictionaries."""
        result = safe_serialize({"outer": {"inner": "value"}})
        assert result is not None
        assert "inner" in result

    def test_list(self):
        """Test serializing a list."""
        result = safe_serialize([1, 2, 3, "test"])
        assert result is not None
        assert "1" in result
        assert '"test"' in result

    def test_max_depth(self):
        """Test that max depth is respected."""
        deeply_nested = {"a": {"b": {"c": {"d": {"e": {"f": "deep"}}}}}}
        result = safe_serialize(deeply_nested, max_depth=2)
        assert result is not None
        assert "max_depth_exceeded" in result

    def test_object_with_dict(self):
        """Test serializing an object with __dict__."""

        class TestObj:
            def __init__(self):
                self.public = "visible"
                self._private = "hidden"

        result = safe_serialize(TestObj())
        assert result is not None
        assert "visible" in result
        assert "hidden" not in result

    def test_object_with_text(self):
        """Test serializing an object with text attribute."""

        class TextObj:
            def __init__(self):
                self.text = "content"

        result = safe_serialize(TextObj())
        assert result is not None
        assert "content" in result


class TestTruncateIfNeeded:
    """Tests for truncate_if_needed function."""

    def test_no_truncation_needed(self):
        """Test that short strings are not truncated."""
        result = truncate_if_needed("short string", max_length=100)
        assert result == "short string"

    def test_truncation_applied(self):
        """Test that long strings are truncated."""
        result = truncate_if_needed("this is a long string", max_length=10)
        assert len(result) == 10
        assert result == "this is a "

    def test_env_var_limit(self):
        """Test truncation using environment variable."""
        with patch.dict(os.environ, {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "5"}):
            result = truncate_if_needed("long string")
            assert len(result) == 5

    def test_invalid_env_var(self):
        """Test handling of invalid environment variable."""
        with patch.dict(
            os.environ, {"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT": "invalid"}
        ):
            result = truncate_if_needed("test string")
            assert result == "test string"


class TestShouldCaptureContent:
    """Tests for should_capture_content function."""

    def test_default_false(self):
        """Test default is False (no capture)."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is not set
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
            assert should_capture_content() is False

    def test_true_values(self):
        """Test various truthy values."""
        for value in ["true", "True", "TRUE", "1", "yes", "on"]:
            with patch.dict(
                os.environ,
                {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": value},
            ):
                assert should_capture_content() is True

    def test_false_values(self):
        """Test various falsy values."""
        for value in ["false", "False", "0", "no", "off", ""]:
            with patch.dict(
                os.environ,
                {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": value},
            ):
                assert should_capture_content() is False


class TestIsInstrumentationEnabled:
    """Tests for is_instrumentation_enabled function."""

    def test_default_true(self):
        """Test default is True (enabled)."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_ENABLE", None)
            assert is_instrumentation_enabled() is True

    def test_disabled(self):
        """Test disabling instrumentation."""
        with patch.dict(os.environ, {"OTEL_INSTRUMENTATION_GENAI_ENABLE": "false"}):
            assert is_instrumentation_enabled() is False


class TestExtractToolInfo:
    """Tests for extract_tool_info function."""

    def test_kwargs_pattern(self):
        """Test extracting from kwargs with 'key' parameter."""
        args = ()
        kwargs = {"key": "my_tool", "arguments": {"arg1": "value1"}}
        name, arguments = extract_tool_info(args, kwargs)
        assert name == "my_tool"
        assert arguments == {"arg1": "value1"}

    def test_positional_args_pattern(self):
        """Test extracting from positional arguments."""
        args = ("tool_name", {"arg": "val"})
        kwargs = {}
        name, arguments = extract_tool_info(args, kwargs)
        assert name == "tool_name"
        assert arguments == {"arg": "val"}

    def test_empty_args(self):
        """Test handling empty arguments."""
        args = ()
        kwargs = {}
        name, arguments = extract_tool_info(args, kwargs)
        assert name == "unknown_tool"
        assert arguments == {}

    def test_only_tool_name(self):
        """Test when only tool name is provided."""
        args = ("single_arg",)
        kwargs = {}
        name, arguments = extract_tool_info(args, kwargs)
        assert name == "single_arg"
        assert arguments == {}


class TestExtractResultContent:
    """Tests for extract_result_content function."""

    def test_none_result(self):
        """Test extracting from None result."""
        assert extract_result_content(None) is None

    def test_simple_result(self):
        """Test extracting from simple result."""
        result = extract_result_content("simple result")
        assert "simple result" in result

    def test_result_with_content(self):
        """Test extracting from result with content attribute."""

        class MockResult:
            class TextContent:
                text = "extracted text"

            content = [TextContent()]

        result = extract_result_content(MockResult())
        assert result is not None
        assert "extracted text" in result

    def test_result_with_dict_items(self):
        """Test extracting from result with dict-like items."""

        class MockItem:
            def __init__(self):
                self.value = "item_value"

        class MockResult:
            content = [MockItem()]

        result = extract_result_content(MockResult())
        assert result is not None
        assert "item_value" in result


class TestDetectClientNetworkTransport:
    """Tests for detect_client_network_transport function."""

    def test_no_transport_attribute(self):
        """Test client with no transport attribute returns default 'pipe'."""
        client = object()  # plain object, no .transport
        assert detect_client_network_transport(client) == "pipe"

    def test_none_transport(self):
        """Test client with None transport returns default 'pipe'."""

        class MockClient:
            transport = None

        assert detect_client_network_transport(MockClient()) == "pipe"

    def test_stdio_transport(self):
        """Test StdioTransport is detected as 'pipe'."""

        class StdioTransport:
            pass

        class MockClient:
            transport = StdioTransport()

        assert detect_client_network_transport(MockClient()) == "pipe"

    def test_python_stdio_transport(self):
        """Test PythonStdioTransport (subclass name) is detected as 'pipe'."""

        class PythonStdioTransport:
            pass

        class MockClient:
            transport = PythonStdioTransport()

        assert detect_client_network_transport(MockClient()) == "pipe"

    def test_sse_transport(self):
        """Test SSETransport is detected as 'tcp'."""

        class SSETransport:
            pass

        class MockClient:
            transport = SSETransport()

        assert detect_client_network_transport(MockClient()) == "tcp"

    def test_streamable_http_transport(self):
        """Test StreamableHttpTransport is detected as 'tcp'."""

        class StreamableHttpTransport:
            pass

        class MockClient:
            transport = StreamableHttpTransport()

        assert detect_client_network_transport(MockClient()) == "tcp"

    def test_fastmcp_transport_inproc(self):
        """Test FastMCPTransport (in-memory) is detected as 'inproc'."""

        class FastMCPTransport:
            pass

        class MockClient:
            transport = FastMCPTransport()

        assert detect_client_network_transport(MockClient()) == "inproc"

    def test_unknown_transport_defaults_to_pipe(self):
        """Test unknown transport type returns 'pipe'."""

        class CustomTransport:
            pass

        class MockClient:
            transport = CustomTransport()

        assert detect_client_network_transport(MockClient()) == "pipe"

    def test_transport_attribute_error(self):
        """Test exception during detection returns 'pipe'."""

        class BadClient:
            @property
            def transport(self):
                raise RuntimeError("broken")

        assert detect_client_network_transport(BadClient()) == "pipe"


class TestDetectServerNetworkTransport:
    """Tests for detect_server_network_transport function."""

    def test_default_returns_pipe(self):
        """Test that default (no context set) returns 'pipe'."""
        # With no ContextVar set and no request_ctx, should default to pipe
        assert detect_server_network_transport() == "pipe"

    def test_http_request_context_returns_tcp(self):
        """Test that an HTTP request context is detected as 'tcp'."""
        from fastmcp.server.context import request_ctx
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.request = MagicMock()
        mock_ctx.request.headers = {"mcp-session-id": "test-123"}

        token = request_ctx.set(mock_ctx)
        try:
            assert detect_server_network_transport() == "tcp"
        finally:
            request_ctx.reset(token)

    def test_no_request_in_context_returns_pipe(self):
        """Test that a request context without HTTP request returns 'pipe'."""
        from fastmcp.server.context import request_ctx
        from unittest.mock import MagicMock

        mock_ctx = MagicMock(spec=[])  # no attributes at all
        token = request_ctx.set(mock_ctx)
        try:
            assert detect_server_network_transport() == "pipe"
        finally:
            request_ctx.reset(token)

    def test_import_error_defaults_to_pipe(self):
        """Test that import error returns 'pipe' gracefully."""
        with patch.dict("sys.modules", {"fastmcp.server.context": None}):
            assert detect_server_network_transport() == "pipe"
