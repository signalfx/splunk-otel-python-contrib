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
    MCPToolCall,
    Workflow,
    Error,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    extract_tool_info,
    extract_result_content,
    should_capture_content,
    truncate_if_needed,
)

_LOGGER = logging.getLogger(__name__)


class ServerInstrumentor:
    """Handles FastMCP server-side instrumentation.

    Instruments:
    - FastMCP.__init__: Capture server name for context
    - ToolManager.call_tool: Trace individual tool executions
    """

    def __init__(self, telemetry_handler: TelemetryHandler):
        self._handler = telemetry_handler
        self._server_name: Optional[str] = None
        self._server_workflow: Optional[Workflow] = None

    def instrument(self):
        """Apply FastMCP server-side instrumentation."""
        # Instrument FastMCP.__init__ to capture server name
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp",
                "FastMCP.__init__",
                self._fastmcp_init_wrapper(),
            ),
            "fastmcp",
        )

        # Instrument ToolManager.call_tool for tool execution
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.tools.tool_manager",
                "ToolManager.call_tool",
                self._tool_call_wrapper(),
            ),
            "fastmcp.tools.tool_manager",
        )

    def uninstrument(self):
        """Remove FastMCP server-side instrumentation.

        Note: wrapt doesn't provide a clean way to unwrap post-import hooks.
        This is a known limitation.
        """
        pass

    def _fastmcp_init_wrapper(self):
        """Create wrapper for FastMCP initialization to capture server name."""
        instrumentor = self

        def traced_init(wrapped, instance, args, kwargs):
            try:
                result = wrapped(*args, **kwargs)

                # Extract server name from args or kwargs
                if args and len(args) > 0:
                    instrumentor._server_name = f"{args[0]}"
                elif "name" in kwargs:
                    instrumentor._server_name = f"{kwargs['name']}"
                else:
                    instrumentor._server_name = "mcp_server"

                _LOGGER.debug(
                    f"FastMCP server initialized: {instrumentor._server_name}"
                )
                return result
            except Exception as e:
                _LOGGER.debug(f"Error in FastMCP init wrapper: {e}", exc_info=True)
                return wrapped(*args, **kwargs)

        return traced_init

    def _tool_call_wrapper(self):
        """Create wrapper for FastMCP tool execution."""
        instrumentor = self
        handler = self._handler

        async def traced_tool_call(wrapped, instance, args, kwargs):
            # Extract tool information
            tool_name, tool_arguments = extract_tool_info(args, kwargs)

            # Create MCPToolCall entity with MCP semantic convention attributes
            tool_call = MCPToolCall(
                name=tool_name,
                arguments=tool_arguments,
                id=str(uuid4()),
                framework="fastmcp",
                system="mcp",
                # Per execute_tool semconv: tool_type indicates type of tool
                # MCP tools are "extension" - executed on agent-side calling external APIs
                tool_type="extension",
                # MCP semantic convention fields
                mcp_method_name="tools/call",  # Per OTel semconv for tool calls
                network_transport="pipe",  # stdio transport = pipe
                mcp_server_name=instrumentor._server_name,  # Server name from init
                is_client=False,  # This is server-side
            )

            # Capture input if content capture is enabled
            # Note: arguments field has semconv_content metadata, will be applied if capture enabled

            # Start tool tracking
            handler.start_tool_call(tool_call)

            start_time = time.time()
            try:
                # Execute the original tool call
                result = await wrapped(*args, **kwargs)

                # Record duration for metrics
                tool_call.duration_s = time.time() - start_time

                # Capture output if content capture is enabled
                if result:
                    try:
                        output_content = extract_result_content(result)
                        if output_content:
                            # Track output size for metrics (impacts LLM context)
                            tool_call.output_size_bytes = len(
                                output_content.encode("utf-8")
                            )
                            if should_capture_content():
                                # Use tool_result field - span emitter applies via semconv_content
                                tool_call.tool_result = truncate_if_needed(
                                    output_content
                                )
                    except Exception as e:
                        _LOGGER.debug(f"Error capturing tool output: {e}")

                # Check for error in result (tool_error per semconv)
                if hasattr(result, "isError") and result.isError:
                    tool_call.is_error = True
                    tool_call.error_type = "tool_error"

                # Stop tool tracking (success)
                handler.stop_tool_call(tool_call)

                return result

            except Exception as e:
                # Record error with duration
                tool_call.duration_s = time.time() - start_time
                tool_call.is_error = True
                tool_call.error_type = type(e).__name__

                # Fail tool tracking
                handler.fail_tool_call(tool_call, Error(type=type(e), message=str(e)))
                raise

        return traced_tool_call
