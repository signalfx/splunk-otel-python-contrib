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
from opentelemetry.util.genai.types import AgentInvocation, MCPToolCall


class TestClientInstrumentor:
    """Tests for ClientInstrumentor class."""

    def test_init(self, mock_telemetry_handler):
        """Test ClientInstrumentor initialization."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        assert instrumentor._handler == mock_telemetry_handler
        assert instrumentor._active_sessions == {}

    @pytest.mark.asyncio
    async def test_client_enter_wrapper(self, mock_telemetry_handler):
        """Test client enter wrapper creates session."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_wrapped = AsyncMock(return_value="session_result")
        mock_instance = MagicMock()

        result = await wrapper(mock_wrapped, mock_instance, (), {})

        assert result == "session_result"
        assert mock_telemetry_handler.start_agent.called

        # Verify session was stored
        assert id(mock_instance) in instrumentor._active_sessions

        # Verify AgentInvocation was created
        agent = mock_telemetry_handler.start_agent.call_args[0][0]
        assert isinstance(agent, AgentInvocation)
        assert agent.name == "mcp.client"
        assert agent.framework == "fastmcp"

    @pytest.mark.asyncio
    async def test_client_exit_wrapper_success(self, mock_telemetry_handler):
        """Test client exit wrapper for successful session end."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)

        # Pre-populate a session
        mock_instance = MagicMock()
        session = AgentInvocation(name="mcp.client")
        instrumentor._active_sessions[id(mock_instance)] = session

        wrapper = instrumentor._client_exit_wrapper()
        mock_wrapped = AsyncMock(return_value=None)

        # Exit with no exception
        await wrapper(mock_wrapped, mock_instance, (None, None, None), {})

        assert mock_telemetry_handler.stop_agent.called
        assert not mock_telemetry_handler.fail_agent.called
        assert id(mock_instance) not in instrumentor._active_sessions

    @pytest.mark.asyncio
    async def test_client_exit_wrapper_with_exception(self, mock_telemetry_handler):
        """Test client exit wrapper when session ends with exception."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)

        mock_instance = MagicMock()
        session = AgentInvocation(name="mcp.client")
        instrumentor._active_sessions[id(mock_instance)] = session

        wrapper = instrumentor._client_exit_wrapper()
        mock_wrapped = AsyncMock(return_value=None)

        # Exit with exception
        await wrapper(
            mock_wrapped,
            mock_instance,
            (ValueError, ValueError("test error"), None),
            {},
        )

        assert mock_telemetry_handler.fail_agent.called
        assert not mock_telemetry_handler.stop_agent.called

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
        """Test client list_tools wrapper."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_list_tools_wrapper()

        # Create mock result with tools
        class MockTool:
            def __init__(self, name):
                self.name = name

        mock_result = MagicMock()
        mock_result.tools = [MockTool("tool1"), MockTool("tool2")]
        mock_wrapped = AsyncMock(return_value=mock_result)

        result = await wrapper(mock_wrapped, MagicMock(), (), {})

        assert result == mock_result
        assert mock_telemetry_handler.start_step.called
        assert mock_telemetry_handler.stop_step.called

        # Verify Step attributes - MCP semantic conventions
        step = mock_telemetry_handler.start_step.call_args[0][0]
        assert step.name == "list_tools"
        assert step.step_type == "admin"
        assert step.attributes["mcp.method.name"] == "tools/list"
        assert step.attributes["network.transport"] == "pipe"
