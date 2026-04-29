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

"""Tests for FastMCP client-side instrumentation."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from opentelemetry.instrumentation.fastmcp.client_instrumentor import (
    ClientInstrumentor,
)
from opentelemetry.util.genai.types import MCPOperation, MCPToolCall


class TestClientInstrumentor:
    """Tests for ClientInstrumentor class."""

    def test_init(self, mock_telemetry_handler):
        """Test ClientInstrumentor initialization."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        assert instrumentor._handler == mock_telemetry_handler
        assert instrumentor._active_sessions == {}

    @pytest.mark.asyncio
    async def test_client_enter_wrapper(self, mock_telemetry_handler):
        """Test client enter wrapper creates initialize MCPOperation session span."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_instance = MagicMock()
        mock_instance.initialize_result = None
        mock_wrapped = AsyncMock(return_value="session_result")

        result = await wrapper(mock_wrapped, mock_instance, (), {})

        assert result == "session_result"
        assert mock_telemetry_handler.start_mcp_operation.called

        # Verify session was stored
        assert id(mock_instance) in instrumentor._active_sessions

        # Verify MCPOperation(initialize) was created
        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert isinstance(init_op, MCPOperation)
        assert init_op.mcp_method_name == "initialize"
        assert init_op.is_client is True
        assert init_op.framework == "fastmcp"
        assert init_op.mcp_session_id is None
        assert init_op.conversation_id is None

    @pytest.mark.asyncio
    async def test_client_enter_wrapper_enriches_protocol_version(
        self, mock_telemetry_handler
    ):
        """Test enter wrapper enriches protocol version from initialize result."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_init_result = MagicMock()
        mock_init_result.protocolVersion = "2025-06-18"
        mock_init_result.serverInfo = MagicMock(name="weather-server")
        mock_instance = MagicMock()
        mock_instance.initialize_result = mock_init_result
        mock_wrapped = AsyncMock(return_value=None)

        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = instrumentor._active_sessions[id(mock_instance)]
        assert init_op.mcp_protocol_version == "2025-06-18"

    @pytest.mark.asyncio
    async def test_client_exit_wrapper_success(self, mock_telemetry_handler):
        """Test client exit wrapper closes the initialize session span on success."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)

        mock_instance = MagicMock()
        init_op = MCPOperation(
            target="",
            mcp_method_name="initialize",
            is_client=True,
            framework="fastmcp",
            system="mcp",
        )
        instrumentor._active_sessions[id(mock_instance)] = init_op

        wrapper = instrumentor._client_exit_wrapper()
        mock_wrapped = AsyncMock(return_value=None)

        await wrapper(mock_wrapped, mock_instance, (None, None, None), {})

        assert mock_telemetry_handler.stop_mcp_operation.called
        assert not mock_telemetry_handler.fail_mcp_operation.called
        assert id(mock_instance) not in instrumentor._active_sessions

    @pytest.mark.asyncio
    async def test_client_exit_wrapper_with_exception(self, mock_telemetry_handler):
        """Test client exit wrapper marks session as failed on exception."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)

        mock_instance = MagicMock()
        init_op = MCPOperation(
            target="",
            mcp_method_name="initialize",
            is_client=True,
            framework="fastmcp",
            system="mcp",
        )
        instrumentor._active_sessions[id(mock_instance)] = init_op

        wrapper = instrumentor._client_exit_wrapper()
        mock_wrapped = AsyncMock(return_value=None)

        await wrapper(
            mock_wrapped,
            mock_instance,
            (ValueError, ValueError("test error"), None),
            {},
        )

        assert mock_telemetry_handler.fail_mcp_operation.called
        assert not mock_telemetry_handler.stop_mcp_operation.called

    @pytest.mark.asyncio
    async def test_client_call_tool_wrapper_success(self, mock_telemetry_handler):
        """Test client call_tool wrapper for successful call."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_call_tool_wrapper()

        mock_result = MagicMock()
        mock_wrapped = AsyncMock(return_value=mock_result)

        result = await wrapper(
            mock_wrapped, MagicMock(), ("my_tool", {"arg": "value"}), {}
        )

        assert result == mock_result
        assert mock_telemetry_handler.start_tool_call.called
        assert mock_telemetry_handler.stop_tool_call.called

        # Verify MCPToolCall was created with MCP fields for metrics
        tool_call = mock_telemetry_handler.start_tool_call.call_args[0][0]
        assert isinstance(tool_call, MCPToolCall)
        assert tool_call.name == "my_tool"
        assert tool_call.arguments == {"arg": "value"}
        assert tool_call.mcp_method_name == "tools/call"
        assert tool_call.network_transport == "pipe"
        assert tool_call.is_client is True

    @pytest.mark.asyncio
    async def test_client_call_tool_wrapper_failure(self, mock_telemetry_handler):
        """Test client call_tool wrapper for failed call."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_call_tool_wrapper()

        mock_wrapped = AsyncMock(side_effect=RuntimeError("Tool failed"))

        with pytest.raises(RuntimeError, match="Tool failed"):
            await wrapper(mock_wrapped, MagicMock(), ("failing_tool",), {})

        assert mock_telemetry_handler.start_tool_call.called
        assert mock_telemetry_handler.fail_tool_call.called
        assert not mock_telemetry_handler.stop_tool_call.called

    @pytest.mark.asyncio
    async def test_client_list_tools_wrapper(self, mock_telemetry_handler):
        """Test client list_tools wrapper uses MCPOperation."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_list_tools_wrapper()

        mock_result = MagicMock()
        mock_result.tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
        mock_wrapped = AsyncMock(return_value=mock_result)

        result = await wrapper(mock_wrapped, MagicMock(), (), {})

        assert result == mock_result
        assert mock_telemetry_handler.start_mcp_operation.called
        assert mock_telemetry_handler.stop_mcp_operation.called

        op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert isinstance(op, MCPOperation)
        assert op.mcp_method_name == "tools/list"
        assert op.is_client is True
        assert op.network_transport == "pipe"
