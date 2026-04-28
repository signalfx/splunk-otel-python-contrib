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

"""Tests for HTTP transport metadata on FastMCP client and server spans.

Validates that:
- network.transport is "tcp" for HTTP transports, "pipe" for stdio
- network.protocol.name / network.protocol.version are set for HTTP
- server.address / server.port populated from client transport URL
- mcp.session.id populated from Mcp-Session-Id header (server) and session (client)
- client.address / client.port populated from Starlette request.client (server)
- error.type is set on failure spans
- duration_s is populated for all MCP operations
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.instrumentation.fastmcp.client_instrumentor import (
    ClientInstrumentor,
    _enrich_client_op,
)
from opentelemetry.instrumentation.fastmcp.server_instrumentor import (
    ServerInstrumentor,
    _enrich_from_request_context,
)
from opentelemetry.instrumentation.fastmcp.transport_instrumentor import (
    TransportInstrumentor,
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
from opentelemetry.util.genai.types import MCPOperation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_http_transport(url: str):
    """Fake HTTP transport with a URL attribute."""
    t = MagicMock()
    t.__class__.__name__ = "StreamableHttpTransport"
    t.url = url
    return t


def _make_stdio_transport():
    t = MagicMock()
    t.__class__.__name__ = "StdioTransport"
    return t


# ---------------------------------------------------------------------------
# utils.py — extract_server_info
# ---------------------------------------------------------------------------


class TestExtractServerInfo:
    def test_parses_http_url_from_transport(self):
        instance = MagicMock()
        instance.transport = MagicMock()
        instance.transport.url = "http://localhost:8080/mcp"
        host, port = extract_server_info(instance)
        assert host == "localhost"
        assert port == 8080

    def test_parses_https_url_default_port(self):
        instance = MagicMock()
        instance.transport = MagicMock()
        instance.transport.url = "https://myserver.example.com/mcp"
        host, port = extract_server_info(instance)
        assert host == "myserver.example.com"
        assert port is None  # no explicit port in URL

    def test_falls_back_to_base_url(self):
        instance = MagicMock()
        instance.transport = MagicMock(spec=[])  # no url attr
        instance.transport.base_url = "http://api.example.com:9000"
        host, port = extract_server_info(instance)
        assert host == "api.example.com"
        assert port == 9000

    def test_returns_none_for_stdio(self):
        instance = MagicMock()
        instance.transport = None
        instance.base_url = None
        instance._base_url = None
        host, port = extract_server_info(instance)
        assert host is None
        assert port is None

    def test_returns_none_when_no_transport(self):
        instance = MagicMock(spec=[])
        host, port = extract_server_info(instance)
        assert host is None
        assert port is None


# ---------------------------------------------------------------------------
# utils.py — extract_session_id
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_reads_session_id_from_session(self):
        instance = MagicMock()
        instance.session.session_id = "sess-abc-123"
        sid = extract_session_id(instance)
        assert sid == "sess-abc-123"

    def test_reads_session_id_from_private_session(self):
        instance = MagicMock()
        instance.session = None
        instance._session = MagicMock()
        instance._session.session_id = "sess-xyz-456"
        sid = extract_session_id(instance)
        assert sid == "sess-xyz-456"

    def test_returns_none_when_no_session(self):
        instance = MagicMock()
        instance.session = None
        instance._session = None
        sid = extract_session_id(instance)
        assert sid is None

    def test_returns_none_when_session_has_no_id(self):
        instance = MagicMock()
        instance.session = MagicMock(spec=[])  # no session_id attr
        instance._session = None
        sid = extract_session_id(instance)
        assert sid is None


# ---------------------------------------------------------------------------
# utils.py — extract_protocol_version
# ---------------------------------------------------------------------------


class TestExtractProtocolVersion:
    def test_reads_version_from_session(self):
        instance = MagicMock()
        instance.session.initialize_result.protocolVersion = "2025-11-25"
        pv = extract_protocol_version(instance)
        assert pv == "2025-11-25"

    def test_returns_none_when_no_session(self):
        instance = MagicMock()
        instance.session = None
        instance._session = None
        pv = extract_protocol_version(instance)
        assert pv is None


# ---------------------------------------------------------------------------
# _enrich_client_op
# ---------------------------------------------------------------------------


class TestEnrichClientOp:
    def test_sets_http_fields_for_tcp_transport(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/call",
            network_transport="tcp",
            is_client=True,
            framework="fastmcp",
            system="mcp",
        )
        instance = MagicMock()
        instance.transport = MagicMock()
        instance.transport.url = "http://127.0.0.1:8080/mcp"
        instance.session = None
        instance._session = None

        _enrich_client_op(op, instance)

        assert op.network_protocol_name == "http"
        assert op.network_protocol_version == "1.1"
        assert op.server_address == "127.0.0.1"
        assert op.server_port == 8080

    def test_no_http_fields_for_pipe_transport(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/call",
            network_transport="pipe",
            is_client=True,
            framework="fastmcp",
            system="mcp",
        )
        instance = MagicMock()
        instance.transport = None
        instance.session = None
        instance._session = None

        _enrich_client_op(op, instance)

        assert op.network_protocol_name is None
        assert op.server_address is None
        assert op.server_port is None

    def test_populates_session_id_when_available(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/list",
            network_transport="tcp",
            is_client=True,
            framework="fastmcp",
            system="mcp",
        )
        instance = MagicMock()
        instance.transport = MagicMock(spec=[])
        instance.session.session_id = "my-session-id"

        _enrich_client_op(op, instance)

        assert op.mcp_session_id == "my-session-id"


# ---------------------------------------------------------------------------
# _enrich_from_request_context (server side)
# ---------------------------------------------------------------------------


class TestEnrichFromRequestContext:
    def setup_method(self):
        clear_mcp_request_context()

    def teardown_method(self):
        clear_mcp_request_context()

    def test_copies_http_fields_to_operation(self):
        ctx = MCPRequestContext(
            jsonrpc_request_id="req-1",
            mcp_method_name="tools/call",
            network_transport="tcp",
            network_protocol_name="http",
            network_protocol_version="1.1",
            client_address="10.0.0.5",
            client_port=54321,
            mcp_session_id="srv-session-abc",
        )
        set_mcp_request_context(ctx)

        op = MCPOperation(
            target="my_tool",
            mcp_method_name="tools/call",
            network_transport=None,
            is_client=False,
            framework="fastmcp",
            system="mcp",
        )
        _enrich_from_request_context(op)

        assert op.jsonrpc_request_id == "req-1"
        assert op.network_transport == "tcp"
        assert op.network_protocol_name == "http"
        assert op.network_protocol_version == "1.1"
        assert op.client_address == "10.0.0.5"
        assert op.client_port == 54321
        assert op.mcp_session_id == "srv-session-abc"

    def test_does_not_overwrite_existing_fields(self):
        ctx = MCPRequestContext(
            network_transport="tcp",
            network_protocol_name="http",
            network_protocol_version="2",
        )
        set_mcp_request_context(ctx)

        op = MCPOperation(
            target="",
            mcp_method_name="tools/call",
            network_transport="pipe",  # already set
            is_client=False,
            framework="fastmcp",
            system="mcp",
        )
        op.network_protocol_name = "grpc"  # already set
        _enrich_from_request_context(op)

        assert op.network_transport == "pipe"  # not overwritten
        assert op.network_protocol_name == "grpc"  # not overwritten

    def test_noop_when_no_context(self):
        op = MCPOperation(
            target="",
            mcp_method_name="tools/call",
            network_transport="pipe",
            is_client=False,
            framework="fastmcp",
            system="mcp",
        )
        _enrich_from_request_context(op)
        # Nothing changed
        assert op.network_protocol_name is None
        assert op.client_address is None


# ---------------------------------------------------------------------------
# transport_instrumentor — HTTP metadata extraction from Starlette scope
# ---------------------------------------------------------------------------


class TestTransportInstrumentorHttpMetadata:
    @pytest.mark.asyncio
    async def test_sets_http_fields_from_starlette_scope_on_tcp(self):
        """HTTP metadata extracted from message.message_metadata.request_context."""
        instrumentor = TransportInstrumentor()
        wrapper_func = instrumentor._server_handle_request_wrapper()

        # Build Starlette-style request_context mock
        mock_scope = {"http_version": "1.1"}
        mock_client = MagicMock()
        mock_client.host = "192.168.1.10"
        mock_client.port = 62000

        mock_starlette_req = MagicMock()
        mock_starlette_req.scope = mock_scope
        mock_starlette_req.client = mock_client
        mock_headers = {"mcp-session-id": "test-session-001"}
        mock_starlette_req.headers = mock_headers

        mock_msg_meta = MagicMock()
        mock_msg_meta.request_context = mock_starlette_req

        mock_message = MagicMock()
        mock_message.request_meta = None
        mock_message.request_id = "rpc-42"
        mock_message.message_metadata = mock_msg_meta

        # TCP instance
        mock_instance = MagicMock()
        mock_instance.transport = MagicMock()
        mock_instance.transport.__class__.__name__ = "streamablehttp"

        from opentelemetry.instrumentation.fastmcp._mcp_context import (
            get_mcp_request_context,
        )

        captured_ctx = {}

        async def capturing_wrapped(*args, **kwargs):
            captured_ctx["ctx"] = get_mcp_request_context()
            return "ok"

        await wrapper_func(capturing_wrapped, mock_instance, (mock_message, None), {})

        ctx = captured_ctx.get("ctx")
        assert ctx is not None
        assert ctx.network_transport == "tcp"
        assert ctx.network_protocol_name == "http"
        assert ctx.network_protocol_version == "1.1"
        assert ctx.client_address == "192.168.1.10"
        assert ctx.client_port == 62000
        assert ctx.mcp_session_id == "test-session-001"

    @pytest.mark.asyncio
    async def test_no_http_fields_for_stdio(self):
        """stdio transport should not set HTTP metadata."""
        instrumentor = TransportInstrumentor()
        wrapper_func = instrumentor._server_handle_request_wrapper()

        mock_message = MagicMock()
        mock_message.request_meta = None
        mock_message.request_id = "rpc-10"

        mock_instance = MagicMock()
        mock_instance.transport = None  # stdio = pipe

        from opentelemetry.instrumentation.fastmcp._mcp_context import (
            get_mcp_request_context,
        )

        captured_ctx = {}

        async def capturing_wrapped(*args, **kwargs):
            captured_ctx["ctx"] = get_mcp_request_context()
            return "ok"

        await wrapper_func(capturing_wrapped, mock_instance, (mock_message, None), {})

        ctx = captured_ctx.get("ctx")
        assert ctx is not None
        assert ctx.network_transport == "pipe"
        assert ctx.network_protocol_name is None
        assert ctx.client_address is None
        assert ctx.mcp_session_id is None


# ---------------------------------------------------------------------------
# client_instrumentor — initialize span HTTP enrichment
# ---------------------------------------------------------------------------


class TestClientInitializeHttpEnrichment:
    @pytest.mark.asyncio
    async def test_initialize_span_enriched_for_http_transport(
        self, mock_telemetry_handler
    ):
        """initialize MCPOperation gets HTTP fields when transport is TCP."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = _make_http_transport("http://localhost:9000/mcp")
        mock_instance.initialize_result = None
        mock_instance.session = None
        mock_instance._session = None

        mock_wrapped = AsyncMock(return_value=None)

        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert init_op.network_transport == "tcp"
        assert init_op.network_protocol_name == "http"
        assert init_op.server_address == "localhost"
        assert init_op.server_port == 9000

    @pytest.mark.asyncio
    async def test_initialize_span_no_http_fields_for_stdio(
        self, mock_telemetry_handler
    ):
        """initialize MCPOperation does not get HTTP fields for stdio."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = _make_stdio_transport()
        mock_instance.initialize_result = None
        mock_instance.session = None
        mock_instance._session = None

        mock_wrapped = AsyncMock(return_value=None)

        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert init_op.network_transport == "pipe"
        assert init_op.network_protocol_name is None
        assert init_op.server_address is None

    @pytest.mark.asyncio
    async def test_initialize_span_has_start_time(self, mock_telemetry_handler):
        """initialize MCPOperation has start_time auto-set by GenAI base class."""
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_enter_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = _make_stdio_transport()
        mock_instance.initialize_result = None
        mock_instance.session = None
        mock_instance._session = None

        mock_wrapped = AsyncMock(return_value=None)
        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        # start_time is set by GenAI base class default_factory; handler sets end_time
        assert init_op.start_time > 0
        assert init_op.duration_s is None  # instrumentation layer does not set this


# ---------------------------------------------------------------------------
# server_instrumentor — initialize span transport detection
# ---------------------------------------------------------------------------


class TestServerInitializeTransportDetection:
    @pytest.mark.asyncio
    async def test_server_run_detects_tcp_transport(self, mock_telemetry_handler):
        """Server initialize span detects HTTP transport from instance."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        instrumentor._server_name = "test-server"
        wrapper = instrumentor._server_run_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = MagicMock()
        mock_instance.transport.__class__.__name__ = "StreamableHttpTransport"

        mock_wrapped = AsyncMock(return_value=None)
        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert init_op.network_transport == "tcp"
        assert init_op.network_protocol_name == "http"

    @pytest.mark.asyncio
    async def test_server_run_defaults_to_pipe_for_stdio(self, mock_telemetry_handler):
        """Server initialize span uses pipe for stdio."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        instrumentor._server_name = "stdio-server"
        wrapper = instrumentor._server_run_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = None

        mock_wrapped = AsyncMock(return_value=None)
        await wrapper(mock_wrapped, mock_instance, (), {})

        init_op = mock_telemetry_handler.start_mcp_operation.call_args[0][0]
        assert init_op.network_transport == "pipe"
        assert init_op.network_protocol_name is None

    @pytest.mark.asyncio
    async def test_server_run_sets_error_type_on_failure(self, mock_telemetry_handler):
        """Server initialize span sets error.type on failure."""
        instrumentor = ServerInstrumentor(mock_telemetry_handler)
        instrumentor._server_name = "error-server"
        wrapper = instrumentor._server_run_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = None
        mock_wrapped = AsyncMock(side_effect=RuntimeError("connection reset"))

        with pytest.raises(RuntimeError):
            await wrapper(mock_wrapped, mock_instance, (), {})

        fail_call = mock_telemetry_handler.fail_mcp_operation.call_args[0]
        failed_op = fail_call[0]
        assert failed_op.error_type == "RuntimeError"
        assert failed_op.is_error is True


# ---------------------------------------------------------------------------
# error_type propagation on client ops
# ---------------------------------------------------------------------------


class TestClientOpErrorType:
    @pytest.mark.asyncio
    async def test_call_tool_sets_error_type_on_failure(self, mock_telemetry_handler):
        instrumentor = ClientInstrumentor(mock_telemetry_handler)
        wrapper = instrumentor._client_call_tool_wrapper()

        mock_instance = MagicMock()
        mock_instance.transport = _make_stdio_transport()
        mock_instance.session = None
        mock_instance._session = None

        mock_wrapped = AsyncMock(side_effect=ValueError("bad args"))

        with pytest.raises(ValueError):
            await wrapper(mock_wrapped, mock_instance, ("my_tool",), {})

        fail_call = mock_telemetry_handler.fail_tool_call.call_args[0]
        tool_call = fail_call[0]
        assert tool_call.error_type == "ValueError"
        assert tool_call.is_error is True
        # duration_s is computed by the handler (end_time - start_time); not set here
        assert tool_call.duration_s is None
