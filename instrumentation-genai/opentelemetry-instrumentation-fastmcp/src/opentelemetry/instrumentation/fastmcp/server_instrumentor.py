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
from typing import Optional
from uuid import uuid4

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    MCPOperation,
    MCPToolCall,
    Workflow,
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


def _enrich_from_request_context(op: MCPOperation) -> None:
    """Copy transport-layer metadata from the ContextVar into an operation."""
    ctx = get_mcp_request_context()
    if ctx is None:
        return
    if ctx.jsonrpc_request_id and op.jsonrpc_request_id is None:
        op.jsonrpc_request_id = ctx.jsonrpc_request_id
    if ctx.network_transport and op.network_transport is None:
        op.network_transport = ctx.network_transport


class ServerInstrumentor:
    """Handles FastMCP server-side instrumentation.

    Instruments:
    - FastMCP.__init__: Capture server name for context
    - Server.run: Track server session lifecycle (mcp.server.session.duration)
    - FastMCP.call_tool / ToolManager.call_tool: Trace tool executions
    - FastMCP.read_resource: Trace resource reads
    - FastMCP.render_prompt: Trace prompt rendering (the MCP ``prompts/get``
      handler; ``get_prompt`` is only the internal lookup)
    """

    def __init__(self, telemetry_handler: TelemetryHandler):
        self._handler = telemetry_handler
        self._server_name: Optional[str] = None
        self._server_workflow: Optional[Workflow] = None

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

        # FastMCP 3.x exposes call_tool/read_resource as public methods;
        # FastMCP 2.x uses ToolManager.call_tool and _call_tool/_read_resource.
        # Use _try_wrap so the instrumentor works across major versions.
        tool_wrapper = self._tool_call_wrapper()
        register_post_import_hook(
            lambda _: self._try_wrap(
                "fastmcp.server.server", "FastMCP.call_tool", tool_wrapper
            ),
            "fastmcp.server.server",
        )
        register_post_import_hook(
            lambda _: self._try_wrap(
                "fastmcp.tools.tool_manager",
                "ToolManager.call_tool",
                tool_wrapper,
            ),
            "fastmcp.tools.tool_manager",
        )

        resource_wrapper = self._read_resource_wrapper()
        register_post_import_hook(
            lambda _: self._try_wrap(
                "fastmcp.server.server",
                "FastMCP.read_resource",
                resource_wrapper,
            ),
            "fastmcp.server.server",
        )

        # FastMCP 3.x uses render_prompt as the MCP prompts/get handler;
        # FastMCP 2.x used get_prompt directly.  Hook both with graceful
        # failure so the instrumentor works across major versions.
        prompt_wrapper = self._render_prompt_wrapper()
        register_post_import_hook(
            lambda _: self._try_wrap(
                "fastmcp.server.server", "FastMCP.render_prompt", prompt_wrapper
            ),
            "fastmcp.server.server",
        )
        register_post_import_hook(
            lambda _: self._try_wrap(
                "fastmcp.server.server", "FastMCP.get_prompt", prompt_wrapper
            ),
            "fastmcp.server.server",
        )

    @staticmethod
    def _try_wrap(module: str, name: str, wrapper) -> None:
        """Attempt to wrap a target; silently skip if it doesn't exist."""
        try:
            wrap_function_wrapper(module, name, wrapper)
        except (ImportError, AttributeError):
            _LOGGER.debug("Skipping wrap %s.%s (not available)", module, name)

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
        """Wrapper for mcp.server.lowlevel.Server.run to track server session.

        Creates an AgentInvocation(agent_type="mcp_server") that spans the
        entire Server.run() lifetime, enabling mcp.server.session.duration
        metric recording via MetricsEmitter.
        """
        instrumentor = self
        handler = self._handler

        async def traced_server_run(wrapped, instance, args, kwargs):
            server_name = instrumentor._server_name or "mcp_server"
            session = AgentInvocation(
                name=f"mcp.server.{server_name}",
                agent_type="mcp_server",
                framework="fastmcp",
                system="mcp",
            )
            session.attributes["gen_ai.operation.name"] = "mcp.server_session"
            session.attributes["network.transport"] = "pipe"  # stdio = pipe

            handler.start_agent(session)
            try:
                result = await wrapped(*args, **kwargs)
                handler.stop_agent(session)
                return result
            except Exception as e:
                session.attributes["error.type"] = type(e).__qualname__
                handler.fail_agent(
                    session,
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
                handler.fail_mcp_operation(op, Error(type=type(e), message=str(e)))
                raise

        return traced_render_prompt

    # Kept for backward compatibility with existing tests
    def _get_prompt_wrapper(self):
        """Deprecated: use _render_prompt_wrapper instead."""
        return self._render_prompt_wrapper()
