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

"""Tests for GAP 5: HTTP transport metadata, error.type, and semconv alignment.

Covers:
- error.type attribute on client/server spans (exception + isError=true)
- network.protocol.name / network.protocol.version for tcp transport
- server.address / server.port on CLIENT spans
- client.address / client.port on SERVER spans via MCPRequestContext
- mcp.session.id on both sides
- mcp.protocol.version on client side
- Transport recording table alignment (stdio vs HTTP)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from opentelemetry.instrumentation.fastmcp.client_instrumentor import (
    ClientInstrumentor,
    _enrich_client_op,
)
from opentelemetry.instrumentation.fastmcp.server_instrumentor import (
    ServerInstrumentor,
    _enrich_from_request_context,
)
from opentelemetry.instrumentation.fastmcp._mcp_context import (
    MCPRequestContext,
    set_mcp_request_context,
    clear_mcp_request_context,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    extract_server_info,
    extract_session_id,
    extract_protocol_version,
)
from opentelemetry.util.genai.types import MCPOperation, MCPToolCall

_PATCH_NO_NATIVE = (
    "opentelemetry.instrumentation.fastmcp.server_instrumentor._has_native_telemetry"
)


# ---- Utility function tests ------------------------------------------------


class TestExtractServerInfo:
    def test_from_transport_url(self):
        instance = MagicMock()
        instance.transport = MagicMock()
        instance.transport.url = "http://localhost:8080/mcp"
        addr, port = extract_server_info(instance)
        assert addr == "localhost"
        assert port == 8080

    def test_from_transport_base_url(self):
        instance = MagicMock()
        transport = MagicMock(spec=[])
        transport.base_url = "https://api.example.com:443/v1"
        instance.transport = transport
        addr, port = extract_server_info(instance)
        assert addr == "api.example.com"
        assert port == 443

    def test_from_instance_base_url(self):
        instance = MagicMock(spec=[])
        instance._base_url = "http://10.0.0.1:3000"
        instance.transport = None
        addr, port = extract_server_info(instance)
        assert addr == "10.0.0.1"
        assert port == 3000

    def test_no_url_returns_none(self):
        instance = MagicMock(spec=[])
        instance.transport = None
        addr, port = extract_server_info(instance)
        assert addr is None
        assert port is None

    def test_url_without_port(self):
        instance = MagicMock()
        instance.transport = MagicMock()
        instance.transport.url = "http://example.com/mcp"
        addr, port = extract_server_info(instance)
        assert addr == "example.com"
        assert port is None

    def test_exception_safety(self):
        instance = MagicMock()
        type(instance).transport = property(
            lambda self: (_ for _ in ()).throw(RuntimeError())
        )
        addr, port = extract_server_info(instance)
        assert addr is None
        assert port is None


class TestExtractSessionId:
    def test_from_session(self):
        instance = MagicMock()
        instance.session = MagicMock()
        instance.session.session_id = "sess-abc-123"
        assert extract_session_id(instance) == "sess-abc-123"

    def test_from_underscore_session(self):
        instance = MagicMock(spec=[])
        instance.session = None
        instance._session = MagicMock()
        instance._session.session_id = "sess-xyz"
        assert extract_session_id(instance) == "sess-xyz"

    def test_no_session(self):
        instance = MagicMock(spec=[])
        instance.session = None
        assert extract_session_id(instance) is None


class TestExtractProtocolVersion:
    def test_from_initialize_result(self):
        instance = MagicMock()
        instance.session = MagicMock()
        instance.session.initialize_result = MagicMock()
        instance.session.initialize_result.protocolVersion = "2025-03-26"
        assert extract_protocol_version(instance) == "2025-03-26"

    def test_no_initialize_result(self):
        instance = MagicMock()
        instance.session = MagicMock()
        instance.session.initialize_result = None
        assert extract_protocol_version(instance) is None

    def test_no_session(self):
        instance = MagicMock(spec=[])
        instance.session = None
        assert extract_protocol_version(instance) is None


# ---- _enrich_client_op tests -----------------------------------------------


class TestEnrichClientOp:
    def _make_tcp_instance(self):
        """Create a mock client with HTTP transport and session."""
        inst = MagicMock()
        inst.transport = MagicMock()
        inst.transport.url = "http://mcpserver.local:9090/mcp"
        inst.session = MagicMock()
        inst.session.session_id = "sid-42"
        inst.session.initialize_result = MagicMock()
        inst.session.initialize_result.protocolVersion = "2025-06-18"
        return inst

    def test_tcp_populates_http_fields(self):
        inst = self._make_tcp_instance()
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="tcp",
            is_client=True,
        )
        _enrich_client_op(op, inst)

        assert op.network_protocol_name == "http"
        assert op.network_protocol_version == "1.1"
        assert op.server_address == "mcpserver.local"
        assert op.server_port == 9090
        assert op.mcp_session_id == "sid-42"
        assert op.mcp_protocol_version == "2025-06-18"

    def test_pipe_skips_http_fields(self):
        inst = MagicMock(spec=[])
        inst.session = None
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="pipe",
            is_client=True,
        )
        _enrich_client_op(op, inst)

        assert op.network_protocol_name is None
        assert op.network_protocol_version is None
        assert op.server_address is None
        assert op.server_port is None

    def test_does_not_overwrite_existing(self):
        inst = self._make_tcp_instance()
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="tcp",
            network_protocol_version="2",
            is_client=True,
        )
        _enrich_client_op(op, inst)
        assert op.network_protocol_version == "2"


# ---- _enrich_from_request_context tests -------------------------------------


class TestEnrichFromRequestContext:
    def test_propagates_http_fields(self):
        ctx = MCPRequestContext(
            jsonrpc_request_id="42",
            network_transport="tcp",
            network_protocol_name="http",
            network_protocol_version="2",
            client_address="192.168.1.50",
            client_port=54321,
            mcp_session_id="srv-session-99",
        )
        set_mcp_request_context(ctx)
        try:
            op = MCPOperation(
                target="",
                mcp_method_name="tools/list",
                is_client=False,
            )
            _enrich_from_request_context(op)

            assert op.jsonrpc_request_id == "42"
            assert op.network_transport == "tcp"
            assert op.network_protocol_name == "http"
            assert op.network_protocol_version == "2"
            assert op.client_address == "192.168.1.50"
            assert op.client_port == 54321
            assert op.mcp_session_id == "srv-session-99"
        finally:
            clear_mcp_request_context()

    def test_pipe_transport_no_http_fields(self):
        ctx = MCPRequestContext(
            jsonrpc_request_id="1",
            network_transport="pipe",
        )
        set_mcp_request_context(ctx)
        try:
            op = MCPOperation(
                target="",
                mcp_method_name="tools/list",
                is_client=False,
            )
            _enrich_from_request_context(op)

            assert op.network_transport == "pipe"
            assert op.network_protocol_name is None
            assert op.client_address is None
        finally:
            clear_mcp_request_context()

    def test_no_context(self):
        clear_mcp_request_context()
        op = MCPOperation(target="", mcp_method_name="tools/list", is_client=False)
        _enrich_from_request_context(op)
        assert op.jsonrpc_request_id is None


# ---- Client error.type tests -----------------------------------------------


class TestClientErrorType:
    @pytest.mark.asyncio
    async def test_call_tool_exception_sets_error_type(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_call_tool_wrapper()

        mock_wrapped = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(ConnectionError):
            await wrapper(mock_wrapped, MagicMock(), ("tool",), {})

        tool_call = mock_telemetry_handler.fail_tool_call.call_args[0][0]
        assert tool_call.error_type == "ConnectionError"
        assert tool_call.is_error is True

    @pytest.mark.asyncio
    async def test_read_resource_exception_sets_error_type(
        self, mock_telemetry_handler
    ):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_read_resource_wrapper()

        mock_wrapped = AsyncMock(side_effect=FileNotFoundError("missing"))

        with pytest.raises(FileNotFoundError):
            await wrapper(mock_wrapped, MagicMock(), ("res://x",), {})

        op = mock_telemetry_handler.fail_mcp_operation.call_args[0][0]
        assert op.error_type == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_get_prompt_exception_sets_error_type(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_get_prompt_wrapper()

        mock_wrapped = AsyncMock(side_effect=KeyError("bad"))

        with pytest.raises(KeyError):
            await wrapper(mock_wrapped, MagicMock(), ("prompt",), {})

        op = mock_telemetry_handler.fail_mcp_operation.call_args[0][0]
        assert op.error_type == "KeyError"

    @pytest.mark.asyncio
    async def test_list_tools_exception_sets_error_type(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_list_tools_wrapper()

        mock_wrapped = AsyncMock(side_effect=TimeoutError("timeout"))

        with pytest.raises(TimeoutError):
            await wrapper(mock_wrapped, MagicMock(), (), {})

        op = mock_telemetry_handler.fail_mcp_operation.call_args[0][0]
        assert op.error_type == "TimeoutError"


# ---- Server error.type tests -----------------------------------------------


class TestServerErrorType:
    @pytest.mark.asyncio
    async def test_tool_call_exception_sets_error_type(self, mock_telemetry_handler):
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._tool_call_wrapper()

            mock_wrapped = AsyncMock(side_effect=ValueError("bad input"))

            with pytest.raises(ValueError):
                await wrapper(mock_wrapped, MagicMock(), ("tool",), {})

            tool_call = mock_telemetry_handler.fail_tool_call.call_args[0][0]
            assert tool_call.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_tool_call_isError_sets_tool_error(self, mock_telemetry_handler):
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._tool_call_wrapper()

            mock_result = MagicMock()
            mock_result.isError = True
            mock_result.content = [MagicMock(text="fail")]
            mock_wrapped = AsyncMock(return_value=mock_result)

            await wrapper(mock_wrapped, MagicMock(), ("tool",), {})

            tool_call = mock_telemetry_handler.stop_tool_call.call_args[0][0]
            assert tool_call.error_type == "tool_error"

    @pytest.mark.asyncio
    async def test_read_resource_exception_sets_error_type(
        self, mock_telemetry_handler
    ):
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._read_resource_wrapper()

            mock_wrapped = AsyncMock(side_effect=RuntimeError("boom"))

            with pytest.raises(RuntimeError):
                await wrapper(mock_wrapped, MagicMock(), ("res://x",), {})

            op = mock_telemetry_handler.fail_mcp_operation.call_args[0][0]
            assert op.error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_render_prompt_exception_sets_error_type(
        self, mock_telemetry_handler
    ):
        with patch(_PATCH_NO_NATIVE, return_value=False):
            instrumentor = ServerInstrumentor(mock_telemetry_handler)
            wrapper = instrumentor._render_prompt_wrapper()

            mock_wrapped = AsyncMock(side_effect=KeyError("nope"))

            with pytest.raises(KeyError):
                await wrapper(mock_wrapped, MagicMock(), ("prompt",), {})

            op = mock_telemetry_handler.fail_mcp_operation.call_args[0][0]
            assert op.error_type == "KeyError"


# ---- Client HTTP metadata enrichment tests ----------------------------------


class TestClientHttpMetadata:
    def _make_http_client(self):
        inst = MagicMock()
        inst.transport = MagicMock()
        inst.transport.url = "http://myhost:7777/mcp"
        inst.session = MagicMock()
        inst.session.session_id = "client-sid"
        inst.session.initialize_result = MagicMock()
        inst.session.initialize_result.protocolVersion = "2025-03-26"
        return inst

    @pytest.mark.asyncio
    async def test_call_tool_tcp_enrichment(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_call_tool_wrapper()
        inst = self._make_http_client()

        with patch(
            "opentelemetry.instrumentation.fastmcp.client_instrumentor.detect_transport",
            return_value="tcp",
        ):
            mock_wrapped = AsyncMock(return_value=MagicMock())
            await wrapper(mock_wrapped, inst, ("tool", {}), {})

        tc = mock_telemetry_handler.start_tool_call.call_args[0][0]
        assert tc.network_transport == "tcp"
        assert tc.network_protocol_name == "http"
        assert tc.server_address == "myhost"
        assert tc.server_port == 7777
        assert tc.mcp_session_id == "client-sid"
        assert tc.mcp_protocol_version == "2025-03-26"

    @pytest.mark.asyncio
    async def test_list_tools_tcp_enrichment(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_list_tools_wrapper()
        inst = self._make_http_client()

        with patch(
            "opentelemetry.instrumentation.fastmcp.client_instrumentor.detect_transport",
            return_value="tcp",
        ):
            mock_wrapped = AsyncMock(return_value=MagicMock())
            await wrapper(mock_wrapped, inst, (), {})

        op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert op.server_address == "myhost"
        assert op.server_port == 7777

    @pytest.mark.asyncio
    async def test_pipe_no_http_enrichment(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_list_tools_wrapper()
        inst = MagicMock(spec=[])
        inst.session = None

        mock_wrapped = AsyncMock(return_value=MagicMock())
        await wrapper(mock_wrapped, inst, (), {})

        op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert op.network_protocol_name is None
        assert op.server_address is None


# ---- Server HTTP metadata via MCPRequestContext tests -----------------------


_PATCH_DETECT_TRANSPORT_SERVER = (
    "opentelemetry.instrumentation.fastmcp.server_instrumentor.detect_transport"
)


class TestServerHttpMetadata:
    @pytest.mark.asyncio
    async def test_read_resource_enriched_from_context(self, mock_telemetry_handler):
        with (
            patch(_PATCH_NO_NATIVE, return_value=False),
            patch(_PATCH_DETECT_TRANSPORT_SERVER, return_value="tcp"),
        ):
            ctx = MCPRequestContext(
                jsonrpc_request_id="99",
                network_transport="tcp",
                network_protocol_name="http",
                network_protocol_version="2",
                client_address="10.0.0.5",
                client_port=12345,
                mcp_session_id="srv-sess-1",
            )
            set_mcp_request_context(ctx)
            try:
                instrumentor = ServerInstrumentor(mock_telemetry_handler)
                wrapper = instrumentor._read_resource_wrapper()

                mock_wrapped = AsyncMock(return_value="data")
                await wrapper(mock_wrapped, MagicMock(), ("res://x",), {})

                op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
                assert op.network_transport == "tcp"
                assert op.network_protocol_name == "http"
                assert op.network_protocol_version == "2"
                assert op.client_address == "10.0.0.5"
                assert op.client_port == 12345
                assert op.mcp_session_id == "srv-sess-1"
            finally:
                clear_mcp_request_context()


# ---- MCPOperation type field tests ------------------------------------------


class TestMCPOperationFields:
    def test_error_type_field_exists_on_mcp_operation(self):
        op = MCPOperation(target="t", mcp_method_name="tools/list")
        assert op.error_type is None
        op.error_type = "SomeError"
        assert op.error_type == "SomeError"

    def test_error_type_inherited_on_mcp_tool_call(self):
        tc = MCPToolCall(name="tool", mcp_method_name="tools/call")
        tc.error_type = "tool_error"
        assert tc.error_type == "tool_error"

    def test_all_http_fields_default_none(self):
        op = MCPOperation(target="", mcp_method_name="tools/list")
        assert op.network_protocol_name is None
        assert op.network_protocol_version is None
        assert op.server_address is None
        assert op.server_port is None
        assert op.client_address is None
        assert op.client_port is None
        assert op.mcp_session_id is None
        assert op.mcp_protocol_version is None
        assert op.rpc_response_status_code is None

    def test_semconv_metadata_on_error_type(self):
        from dataclasses import fields as df

        for f in df(MCPOperation):
            if f.name == "error_type":
                assert f.metadata.get("semconv") == "error.type"
                return
        pytest.fail("error_type field not found on MCPOperation")


# ---- Transport recording table alignment ------------------------------------


class TestTransportRecordingTable:
    """Validate the transport recording reference table from the plan.

    | stdio         | pipe | (not set) | (not set) |
    | Streamable HTTP | tcp  | http      | 2 or 1.1  |
    """

    def test_stdio_attributes(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="pipe",
            is_client=True,
        )
        inst = MagicMock(spec=[])
        inst.session = None
        _enrich_client_op(op, inst)

        assert op.network_transport == "pipe"
        assert op.network_protocol_name is None
        assert op.network_protocol_version is None

    def test_http_attributes(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="tcp",
            is_client=True,
        )
        inst = MagicMock()
        inst.transport = MagicMock()
        inst.transport.url = "http://host:8080/mcp"
        inst.session = None
        _enrich_client_op(op, inst)

        assert op.network_transport == "tcp"
        assert op.network_protocol_name == "http"
        assert op.network_protocol_version == "1.1"
