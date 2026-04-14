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
import time
from typing import Optional
from uuid import uuid4

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    MCPOperation,
    MCPToolCall,
    Workflow,
)
from opentelemetry.instrumentation.fastmcp._mcp_context import (
    get_mcp_request_context,
)
from opentelemetry.instrumentation.fastmcp.utils import (
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
    if ctx.network_protocol_name and op.network_protocol_name is None:
        op.network_protocol_name = ctx.network_protocol_name
    if ctx.network_protocol_version and op.network_protocol_version is None:
        op.network_protocol_version = ctx.network_protocol_version
    if ctx.client_address and op.client_address is None:
        op.client_address = ctx.client_address
    if ctx.client_port and op.client_port is None:
        op.client_port = ctx.client_port
    if ctx.server_address and op.server_address is None:
        op.server_address = ctx.server_address
    if ctx.server_port and op.server_port is None:
        op.server_port = ctx.server_port


class ServerInstrumentor:
    """Handles FastMCP server-side instrumentation.

    Instruments:
    - FastMCP.__init__: Capture server name for context
    - FastMCP.call_tool: Trace individual tool executions
    - FastMCP.read_resource: Trace resource reads
    - FastMCP.get_prompt: Trace prompt gets
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

        # Hook FastMCP.call_tool on the server object directly.
        # Older fastmcp versions had ToolManager.call_tool; we hook
        # both paths so the first that imports wins.
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
                "fastmcp.tools.tool_manager",
                "ToolManager.call_tool",
                self._tool_call_wrapper(),
            ),
            "fastmcp.tools.tool_manager",
        )

        # Resources & prompts
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
                "FastMCP.get_prompt",
                self._get_prompt_wrapper(),
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

    # ------------------------------------------------------------------
    # tools/call
    # ------------------------------------------------------------------
    def _tool_call_wrapper(self):
        """Create wrapper for FastMCP tool execution."""
        instrumentor = self
        handler = self._handler

        async def traced_tool_call(wrapped, instance, args, kwargs):
            tool_name, tool_arguments = extract_tool_info(args, kwargs)

            tool_call = MCPToolCall(
                name=tool_name,
                arguments=tool_arguments,
                id=str(uuid4()),
                framework="fastmcp",
                system="mcp",
                tool_type="extension",
                mcp_method_name="tools/call",
                network_transport="pipe",
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
            )
            _enrich_from_request_context(tool_call)

            handler.start_tool_call(tool_call)

            start_time = time.time()
            try:
                result = await wrapped(*args, **kwargs)

                tool_call.duration_s = time.time() - start_time

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
                tool_call.duration_s = time.time() - start_time
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
            uri = ""
            if args:
                uri = str(args[0])
            elif "uri" in kwargs:
                uri = str(kwargs["uri"])

            op = MCPOperation(
                target=uri,
                mcp_method_name="resources/read",
                network_transport="pipe",
                mcp_resource_uri=uri or None,
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
                framework="fastmcp",
                system="mcp",
            )
            _enrich_from_request_context(op)

            handler.start_mcp_operation(op)

            start_time = time.time()
            try:
                result = await wrapped(*args, **kwargs)
                op.duration_s = time.time() - start_time
                handler.stop_mcp_operation(op)
                return result
            except Exception as e:
                op.duration_s = time.time() - start_time
                op.is_error = True
                handler.fail_mcp_operation(op, Error(type=type(e), message=str(e)))
                raise

        return traced_read_resource

    # ------------------------------------------------------------------
    # prompts/get
    # ------------------------------------------------------------------
    def _get_prompt_wrapper(self):
        """Wrapper for FastMCP.get_prompt."""
        instrumentor = self
        handler = self._handler

        async def traced_get_prompt(wrapped, instance, args, kwargs):
            prompt_name = ""
            if args:
                prompt_name = str(args[0])
            elif "name" in kwargs:
                prompt_name = str(kwargs["name"])

            op = MCPOperation(
                target=prompt_name,
                mcp_method_name="prompts/get",
                network_transport="pipe",
                gen_ai_prompt_name=prompt_name or None,
                sdot_mcp_server_name=instrumentor._server_name,
                is_client=False,
                framework="fastmcp",
                system="mcp",
            )
            _enrich_from_request_context(op)

            handler.start_mcp_operation(op)

            start_time = time.time()
            try:
                result = await wrapped(*args, **kwargs)
                op.duration_s = time.time() - start_time
                handler.stop_mcp_operation(op)
                return result
            except Exception as e:
                op.duration_s = time.time() - start_time
                op.is_error = True
                handler.fail_mcp_operation(op, Error(type=type(e), message=str(e)))
                raise

        return traced_get_prompt
