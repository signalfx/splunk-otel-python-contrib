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

"""Tests for FastMCP server-side instrumentation."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from opentelemetry.instrumentation.fastmcp.server_instrumentor import (
    ServerInstrumentor,
)
from opentelemetry.util.genai.types import MCPToolCall


class TestServerInstrumentor:
    """Tests for ServerInstrumentor class."""

    def test_init(self, mock_telemetry_handler):
        """Test ServerInstrumentor initialization."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        assert instrumentor._handler == mock_telemetry_handler
        assert instrumentor._server_name is None

    def test_fastmcp_init_wrapper_with_args(self, mock_telemetry_handler):
        """Test that FastMCP init wrapper captures server name from args."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._fastmcp_init_wrapper()

        # Mock the wrapped function
        mock_wrapped = MagicMock(return_value=None)
        mock_instance = MagicMock()

        # Call with positional arg
        wrapper(mock_wrapped, mock_instance, ("my-server",), {})

        assert instrumentor._server_name == "my-server"
        mock_wrapped.assert_called_once()

    def test_fastmcp_init_wrapper_with_kwargs(self, mock_telemetry_handler):
        """Test that FastMCP init wrapper captures server name from kwargs."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._fastmcp_init_wrapper()

        mock_wrapped = MagicMock(return_value=None)
        mock_instance = MagicMock()

        wrapper(mock_wrapped, mock_instance, (), {"name": "kwargs-server"})

        assert instrumentor._server_name == "kwargs-server"

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_success(self, mock_telemetry_handler):
        """Test tool call wrapper for successful execution."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        instrumentor._server_name = "test-server"
        wrapper = instrumentor._tool_call_wrapper()

        # Mock the wrapped async function
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="result text")]
        mock_result.isError = False
        mock_wrapped = AsyncMock(return_value=mock_result)
        mock_instance = MagicMock()

        # Call the wrapper
        result = await wrapper(
            mock_wrapped, mock_instance, ("my_tool", {"arg1": "value1"}), {}
        )

        # Verify handler was called
        assert mock_telemetry_handler.start_tool_call.called
        assert mock_telemetry_handler.stop_tool_call.called
        assert not mock_telemetry_handler.fail_tool_call.called

        # Verify result is returned
        assert result == mock_result

        # Verify MCPToolCall was created with correct attributes
        tool_call = mock_telemetry_handler.start_tool_call.call_args[0][0]
        assert isinstance(tool_call, MCPToolCall)
        assert tool_call.name == "my_tool"
        assert tool_call.framework == "fastmcp"
        assert tool_call.system == "mcp"
        assert tool_call.mcp_server_name == "test-server"

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_failure(self, mock_telemetry_handler):
        """Test tool call wrapper for failed execution."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        instrumentor._server_name = "test-server"
        wrapper = instrumentor._tool_call_wrapper()

        # Mock the wrapped async function that raises
        mock_wrapped = AsyncMock(side_effect=ValueError("Test error"))
        mock_instance = MagicMock()

        # Call and expect exception
        with pytest.raises(ValueError, match="Test error"):
            await wrapper(
                mock_wrapped, mock_instance, ("failing_tool",), {"arguments": {}}
            )

        # Verify fail_tool_call was called
        assert mock_telemetry_handler.start_tool_call.called
        assert mock_telemetry_handler.fail_tool_call.called
        assert not mock_telemetry_handler.stop_tool_call.called

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_with_content_capture(self, mock_telemetry_handler):
        """Test tool call wrapper with content capture enabled."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._tool_call_wrapper()

        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="captured output")]
        mock_wrapped = AsyncMock(return_value=mock_result)

        with patch.dict(
            "os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}
        ):
            await wrapper(
                mock_wrapped, MagicMock(), ("tool_name", {"input": "data"}), {}
            )

        # Verify the call succeeded and tool was properly tracked
        assert mock_telemetry_handler.start_tool_call.called
        assert mock_telemetry_handler.stop_tool_call.called

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_with_error_result(self, mock_telemetry_handler):
        """Test tool call wrapper when result indicates an error."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._tool_call_wrapper()

        # Create result with isError=True
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [MagicMock(text="error message")]
        mock_wrapped = AsyncMock(return_value=mock_result)

        result = await wrapper(
            mock_wrapped, MagicMock(), ("error_tool",), {"arguments": {}}
        )

        # Result should still be returned
        assert result == mock_result

        # stop_tool_call should be called (not fail_tool_call since no exception)
        assert mock_telemetry_handler.stop_tool_call.called

        # Verify error_type field was set per MCP semconv
        tool_call = mock_telemetry_handler.stop_tool_call.call_args[0][0]
        assert tool_call.error_type == "tool_error"
