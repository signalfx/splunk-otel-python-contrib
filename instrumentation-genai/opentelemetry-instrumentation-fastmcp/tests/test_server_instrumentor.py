"""Tests for FastMCP server-side instrumentation."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from opentelemetry.instrumentation.fastmcp.server_instrumentor import (
    ServerInstrumentor,
)
from opentelemetry.util.genai.types import MCPOperation, MCPToolCall

_PATCH_NO_NATIVE = (
    "opentelemetry.instrumentation.fastmcp.server_instrumentor._has_native_telemetry"
)


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

        mock_wrapped = MagicMock(return_value=None)
        mock_instance = MagicMock()

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

    # ------------------------------------------------------------------
    # tools/call
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_tool_call_wrapper_success(self, mock_telemetry_handler):
        """Test tool call wrapper for successful execution."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            instrumentor._server_name = "test-server"
            wrapper = instrumentor._tool_call_wrapper()

            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="result text")]
            mock_result.isError = False
            mock_wrapped = AsyncMock(return_value=mock_result)
            mock_instance = MagicMock()

            result = await wrapper(
                mock_wrapped, mock_instance, ("my_tool", {"arg1": "value1"}), {}
            )

            assert mock_telemetry_handler.start_tool_call.called
            assert mock_telemetry_handler.stop_tool_call.called
            assert not mock_telemetry_handler.fail_tool_call.called
            assert result == mock_result

            tool_call = mock_telemetry_handler.start_tool_call.call_args[0][0]
            assert isinstance(tool_call, MCPToolCall)
            assert tool_call.name == "my_tool"
            assert tool_call.framework == "fastmcp"
            assert tool_call.system == "mcp"
            assert tool_call.sdot_mcp_server_name == "test-server"

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_failure(self, mock_telemetry_handler):
        """Test tool call wrapper for failed execution."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            instrumentor._server_name = "test-server"
            wrapper = instrumentor._tool_call_wrapper()

            mock_wrapped = AsyncMock(side_effect=ValueError("Test error"))
            mock_instance = MagicMock()

            with pytest.raises(ValueError, match="Test error"):
                await wrapper(
                    mock_wrapped,
                    mock_instance,
                    ("failing_tool",),
                    {"arguments": {}},
                )

            assert mock_telemetry_handler.start_tool_call.called
            assert mock_telemetry_handler.fail_tool_call.called
            assert not mock_telemetry_handler.stop_tool_call.called

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_with_content_capture(self, mock_telemetry_handler):
        """Test tool call wrapper with content capture enabled."""
        with (
            patch(_PATCH_NO_NATIVE, return_value=False),
            patch.dict(
                "os.environ",
                {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
            ),
        ):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._tool_call_wrapper()

            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="captured output")]
            mock_wrapped = AsyncMock(return_value=mock_result)

            await wrapper(
                mock_wrapped, MagicMock(), ("tool_name", {"input": "data"}), {}
            )

            assert mock_telemetry_handler.start_tool_call.called
            assert mock_telemetry_handler.stop_tool_call.called

    @pytest.mark.asyncio
    async def test_tool_call_wrapper_with_error_result(self, mock_telemetry_handler):
        """Test tool call wrapper when result indicates an error."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._tool_call_wrapper()

            mock_result = MagicMock()
            mock_result.isError = True
            mock_result.content = [MagicMock(text="error message")]
            mock_wrapped = AsyncMock(return_value=mock_result)

            result = await wrapper(
                mock_wrapped, MagicMock(), ("error_tool",), {"arguments": {}}
            )

            assert result == mock_result
            assert mock_telemetry_handler.stop_tool_call.called

            tool_call = mock_telemetry_handler.stop_tool_call.call_args[0][0]
            assert tool_call.error_type == "tool_error"

    # ------------------------------------------------------------------
    # resources/read
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_resource_wrapper_success(self, mock_telemetry_handler):
        """Test read_resource wrapper creates correct MCPOperation."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            instrumentor._server_name = "res-server"
            wrapper = instrumentor._read_resource_wrapper()

            mock_wrapped = AsyncMock(return_value="resource content")
            result = await wrapper(mock_wrapped, MagicMock(), ("system://info",), {})

            assert result == "resource content"
            assert mock_telemetry_handler.start_mcp_operation.called
            assert mock_telemetry_handler.stop_mcp_operation.called

            op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
            assert isinstance(op, MCPOperation)
            assert op.mcp_method_name == "resources/read"
            assert op.target == "system://info"
            assert op.mcp_resource_uri == "system://info"
            assert op.sdot_mcp_server_name == "res-server"
            assert op.is_client is False

    @pytest.mark.asyncio
    async def test_read_resource_wrapper_failure(self, mock_telemetry_handler):
        """Test read_resource wrapper handles exceptions."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._read_resource_wrapper()

            mock_wrapped = AsyncMock(side_effect=FileNotFoundError("not found"))

            with pytest.raises(FileNotFoundError):
                await wrapper(mock_wrapped, MagicMock(), ("system://missing",), {})

            assert mock_telemetry_handler.start_mcp_operation.called
            assert mock_telemetry_handler.fail_mcp_operation.called

    # ------------------------------------------------------------------
    # prompts/get  (render_prompt)
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_render_prompt_wrapper_success(self, mock_telemetry_handler):
        """Test render_prompt wrapper creates correct MCPOperation."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            instrumentor._server_name = "prompt-server"
            wrapper = instrumentor._render_prompt_wrapper()

            mock_wrapped = AsyncMock(return_value="rendered prompt")
            result = await wrapper(mock_wrapped, MagicMock(), ("weather_forecast",), {})

            assert result == "rendered prompt"
            assert mock_telemetry_handler.start_mcp_operation.called
            assert mock_telemetry_handler.stop_mcp_operation.called

            op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
            assert isinstance(op, MCPOperation)
            assert op.mcp_method_name == "prompts/get"
            assert op.target == "weather_forecast"
            assert op.gen_ai_prompt_name == "weather_forecast"
            assert op.sdot_mcp_server_name == "prompt-server"
            assert op.is_client is False

    @pytest.mark.asyncio
    async def test_render_prompt_wrapper_failure(self, mock_telemetry_handler):
        """Test render_prompt wrapper handles exceptions."""
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._render_prompt_wrapper()

            mock_wrapped = AsyncMock(side_effect=KeyError("no such prompt"))

            with pytest.raises(KeyError):
                await wrapper(mock_wrapped, MagicMock(), ("nonexistent",), {})

            assert mock_telemetry_handler.start_mcp_operation.called
            assert mock_telemetry_handler.fail_mcp_operation.called

    # ------------------------------------------------------------------
    # Native telemetry dedupe
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_tool_call_skips_when_native_telemetry(self, mock_telemetry_handler):
        """Server wrappers pass through when FastMCP has native telemetry."""
        with patch(_PATCH_NO_NATIVE, return_value=True):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._tool_call_wrapper()

            mock_wrapped = AsyncMock(return_value="native result")
            result = await wrapper(mock_wrapped, MagicMock(), ("tool",), {})

            assert result == "native result"
            assert not mock_telemetry_handler.start_tool_call.called

    @pytest.mark.asyncio
    async def test_read_resource_skips_when_native_telemetry(
        self, mock_telemetry_handler
    ):
        """Server wrappers pass through when FastMCP has native telemetry."""
        with patch(_PATCH_NO_NATIVE, return_value=True):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._read_resource_wrapper()

            mock_wrapped = AsyncMock(return_value="native result")
            result = await wrapper(mock_wrapped, MagicMock(), ("system://info",), {})

            assert result == "native result"
            assert not mock_telemetry_handler.start_mcp_operation.called

    @pytest.mark.asyncio
    async def test_render_prompt_skips_when_native_telemetry(
        self, mock_telemetry_handler
    ):
        """Server wrappers pass through when FastMCP has native telemetry."""
        with patch(_PATCH_NO_NATIVE, return_value=True):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._render_prompt_wrapper()

            mock_wrapped = AsyncMock(return_value="native result")
            result = await wrapper(mock_wrapped, MagicMock(), ("weather",), {})

            assert result == "native result"
            assert not mock_telemetry_handler.start_mcp_operation.called
