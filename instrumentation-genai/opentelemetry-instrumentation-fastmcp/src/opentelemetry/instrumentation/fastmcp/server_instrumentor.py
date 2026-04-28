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

"""FastMCP server-side instrumentation."""

import logging
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    MCPOperation,
    MCPToolCall,
)
from opentelemetry.instrumentation.fastmcp._mcp_context import (
    get_mcp_request_context,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    detect_transport,
    extract_tool_info,
    extract_result_content,
    should_capture_content,
    truncate_if_needed,
)

_LOGGER = logging.getLogger(__name__)

# FastMCP 3.x call_tool recurses through middleware (run_middleware=True →
# middleware → self.call_tool(..., run_middleware=False)).  This guard
# ensures only the outermost call creates a span.
_IN_TOOL_CALL: ContextVar[bool] = ContextVar("_in_tool_call", default=False)


def _enrich_from_request_context(op: MCPOperation) -> None:
    """Copy transport-layer metadata from the ContextVar into an operation."""
    ctx = get_mcp_request_context()
    if ctx is None:
        return
    if ctx.jsonrpc_request_id and op.jsonrpc_request_id is None:
        op.jsonrpc_request_id = ctx.jsonrpc_request_id
    if ctx.network_transport and op.network_transport is None:
        op.network_transport = ctx.network_transport
    if ctx.network_protocol_name and op.network_protocol_name is None:
        op.network_protocol_name = ctx.network_protocol_name
    if ctx.network_protocol_version and op.network_protocol_version is None:
        op.network_protocol_version = ctx.network_protocol_version
    if ctx.client_address and op.client_address is None:
        op.client_address = ctx.client_address
    if ctx.client_port and op.client_port is None:
        op.client_port = ctx.client_port
    if ctx.mcp_session_id and op.mcp_session_id is None:
        op.mcp_session_id = ctx.mcp_session_id


class ServerInstrumentor:
    """Handles FastMCP 3.x server-side instrumentation.

    Instruments:
    - FastMCP.__init__: Capture server name for context
    - Server.run: Track server session lifecycle (mcp.server.session.duration)
    - FastMCP.call_tool: Trace tool executions
    - FastMCP.read_resource: Trace resource reads
    - FastMCP.render_prompt: Trace prompt rendering
    """

    def __init__(self, telemetry_handler: TelemetryHandler):
        self._handler = telemetry_handler
        self._server_name: Optional[str] = None

    def instrument(self):
        """Apply FastMCP server-side instrumentation."""
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp",
                "FastMCP.__init__",
                self._fastmcp_init_wrapper(),
            ),
            "fastmcp",
        )

        # Instrument Server.run to track server session lifecycle
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.lowlevel.server",
                "Server.run",
                self._server_run_wrapper(),
            ),
            "mcp.server.lowlevel.server",
        )

        # Wrap FastMCP server methods (3.x API surface).
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.server.server",
                "FastMCP.call_tool",
                self._tool_call_wrapper(),
            ),
            "fastmcp.server.server",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.server.server",
                "FastMCP.read_resource",
                self._read_resource_wrapper(),
            ),
            "fastmcp.server.server",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.server.server",
                "FastMCP.render_prompt",
                self._render_prompt_wrapper(),
            ),
            "fastmcp.server.server",
        )

    def uninstrument(self):
        """Remove FastMCP server-side instrumentation.

        Note: wrapt doesn't provide a clean way to unwrap post-import hooks.
        """
        pass

    def _fastmcp_init_wrapper(self):
        """Create wrapper for FastMCP initialization to capture server name."""
        instrumentor = self

        def traced_init(wrapped, instance, args, kwargs):
            try:
                result = wrapped(*args, **kwargs)

                if args and len(args) > 0:
                    instrumentor._server_name = f"{args[0]}"
                elif "name" in kwargs:
                    instrumentor._server_name = f"{kwargs['name']}"
                else:
                    instrumentor._server_name = "mcp_server"

                _LOGGER.debug(
                    "FastMCP server initialized: %s",
                    instrumentor._server_name,
                )
                return result
            except Exception as e:
                _LOGGER.debug("Error in FastMCP init wrapper: %s", e, exc_info=True)
                return wrapped(*args, **kwargs)

        return traced_init

    def _server_run_wrapper(self):
        """Wrapper for mcp.server.lowlevel.Server.run — creates an ``initialize`` root span.

        The span spans the entire Server.run() lifetime so that all server-side
        MCP operations (tools/call, resources/read, prompts/get) are children.
        This mirrors the client's session-spanning ``initialize`` span and
        ensures ``mcp.server.session.duration`` is recorded.

        """
        instrumentor = self
        handler = self._handler

        async def traced_server_run(wrapped, instance, args, kwargs):
            server_name = instrumentor._server_name or "mcp_server"
            transport = detect_transport(instance)

            init_op = MCPOperation(
                target="",
                mcp_method_name="initialize",
                network_transport=transport,
                sdot_mcp_server_name=server_name,
                is_client=False,
                framework="fastmcp",
                system="mcp",
            )
            if transport == "tcp":
                init_op.network_protocol_name = "http"

            handler.start_mcp_operation(init_op)
            try:
                result = await wrapped(*args, **kwargs)
                handler.stop_mcp_operation(init_op)
                return result
            except Exception as e:
                init_op.is_error = True
                init_op.error_type = type(e).__name__
                init_op.mcp_error_type = type(e).__qualname__
                handler.fail_mcp_operation(
                    init_op,
                    Error(type=type(e), message=str(e)),
                )
                raise

        return traced_server_run

    # ------------------------------------------------------------------
    # tools/call
    # ------------------------------------------------------------------
    def _tool_call_wrapper(self):
        """Create wrapper for FastMCP tool execution."""
        instrumentor = self
        handler = self._handler

        async def traced_tool_call(wrapped, instance, args, kwargs):
            if _IN_TOOL_CALL.get():
                return await wrapped(*args, **kwargs)

            tool_name, tool_arguments = extract_tool_info(args, kwargs)
            transport = detect_transport(instance)

            tool_call = MCPToolCall(
                name=tool_name,
                arguments=tool_arguments,
                id=str(uuid4()),
                framework="fastmcp",
                system="mcp",
                tool_type="extension",
                mcp_method_name="tools/call",
                network_transport=transport,
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
            )
            _enrich_from_request_context(tool_call)

            handler.start_tool_call(tool_call)
            token = _IN_TOOL_CALL.set(True)

            try:
                result = await wrapped(*args, **kwargs)

                if result:
                    try:
                        output_content = extract_result_content(result)
                        if output_content:
                            tool_call.output_size_bytes = len(
                                output_content.encode("utf-8")
                            )
                            if should_capture_content():
                                tool_call.tool_result = truncate_if_needed(
                                    output_content
                                )
                    except Exception as e:
                        _LOGGER.debug("Error capturing tool output: %s", e)

                if hasattr(result, "isError") and result.isError:
                    tool_call.is_error = True
                    tool_call.error_type = "tool_error"

                handler.stop_tool_call(tool_call)
                return result

            except Exception as e:
                tool_call.is_error = True
                tool_call.error_type = type(e).__name__
                handler.fail_tool_call(tool_call, Error(type=type(e), message=str(e)))
                raise
            finally:
                _IN_TOOL_CALL.reset(token)

        return traced_tool_call

    # ------------------------------------------------------------------
    # resources/read
    # ------------------------------------------------------------------
    def _read_resource_wrapper(self):
        """Wrapper for FastMCP.read_resource."""
        instrumentor = self
        handler = self._handler

        async def traced_read_resource(wrapped, instance, args, kwargs):
            uri = str(args[0]) if args else str(kwargs.get("uri", ""))
            transport = detect_transport(instance)

            op = MCPOperation(
                target=uri,
                mcp_method_name="resources/read",
                network_transport=transport,
                mcp_resource_uri=uri or None,
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
                framework="fastmcp",
                system="mcp",
            )
            _enrich_from_request_context(op)

            handler.start_mcp_operation(op)

            try:
                result = await wrapped(*args, **kwargs)
                handler.stop_mcp_operation(op)
                return result
            except Exception as e:
                op.is_error = True
                op.error_type = type(e).__name__
                handler.fail_mcp_operation(op, Error(type=type(e), message=str(e)))
                raise

        return traced_read_resource

    # ------------------------------------------------------------------
    # prompts/get  (maps to FastMCP.render_prompt, not get_prompt)
    # ------------------------------------------------------------------
    def _render_prompt_wrapper(self):
        """Wrapper for FastMCP.render_prompt.

        ``render_prompt`` is the server-side handler for the MCP
        ``prompts/get`` protocol method.  ``get_prompt`` is only the
        internal prompt definition lookup and is never invoked by
        the MCP request path.
        """
        instrumentor = self
        handler = self._handler

        async def traced_render_prompt(wrapped, instance, args, kwargs):
            prompt_name = str(args[0]) if args else str(kwargs.get("name", ""))
            transport = detect_transport(instance)

            op = MCPOperation(
                target=prompt_name,
                mcp_method_name="prompts/get",
                network_transport=transport,
                gen_ai_prompt_name=prompt_name or None,
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
                framework="fastmcp",
                system="mcp",
            )
            _enrich_from_request_context(op)

            handler.start_mcp_operation(op)

            try:
                result = await wrapped(*args, **kwargs)
                handler.stop_mcp_operation(op)
                return result
            except Exception as e:
                op.is_error = True
                op.error_type = type(e).__name__
                handler.fail_mcp_operation(op, Error(type=type(e), message=str(e)))
                raise

        return traced_render_prompt

    # Kept for backward compatibility with existing tests
    def _get_prompt_wrapper(self):
        """Deprecated: use _render_prompt_wrapper instead."""
        return self._render_prompt_wrapper()
