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

"""FastMCP client-side instrumentation."""

import logging
import time

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Step,
    Error,
    MCPToolCall,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    safe_serialize,
    should_capture_content,
    truncate_if_needed,
)

_LOGGER = logging.getLogger(__name__)


class ClientInstrumentor:
    """Handles FastMCP client-side instrumentation.

    Instruments:
    - FastMCP Client.__aenter__: Start client session trace
    - FastMCP Client.__aexit__: End client session trace
    - MCP client operations (list_tools, call_tool)
    """

    def __init__(self, telemetry_handler: TelemetryHandler):
        self._handler = telemetry_handler
        self._active_sessions: dict[int, AgentInvocation] = {}

    def instrument(self):
        """Apply FastMCP client-side instrumentation."""
        # Instrument FastMCP Client session lifecycle
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aenter__",
                self._client_enter_wrapper(),
            ),
            "fastmcp.client",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aexit__",
                self._client_exit_wrapper(),
            ),
            "fastmcp.client",
        )

        # Instrument client tool operations
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.call_tool",
                self._client_call_tool_wrapper(),
            ),
            "fastmcp.client",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.list_tools",
                self._client_list_tools_wrapper(),
            ),
            "fastmcp.client",
        )

    def uninstrument(self):
        """Remove FastMCP client-side instrumentation.

        Note: wrapt doesn't provide a clean way to unwrap post-import hooks.
        This is a known limitation.
        """
        pass

    def _client_enter_wrapper(self):
        """Wrapper for FastMCP Client.__aenter__ to start a session trace."""
        instrumentor = self
        handler = self._handler

        async def traced_enter(wrapped, instance, args, kwargs):
            try:
                # Call original
                result = await wrapped(*args, **kwargs)

                # Create an AgentInvocation to represent the client session
                session = AgentInvocation(
                    name="mcp.client",
                    agent_type="mcp_client",
                    framework="fastmcp",
                    system="mcp",
                )
                session.attributes["gen_ai.operation.name"] = "mcp.client_session"

                # Store session by instance id
                instrumentor._active_sessions[id(instance)] = session

                # Start agent invocation
                handler.start_agent(session)

                return result
            except Exception as e:
                _LOGGER.debug(f"Error in client enter wrapper: {e}", exc_info=True)
                return await wrapped(*args, **kwargs)

        return traced_enter

    def _client_exit_wrapper(self):
        """Wrapper for FastMCP Client.__aexit__ to end the session trace."""
        instrumentor = self
        handler = self._handler

        async def traced_exit(wrapped, instance, args, kwargs):
            try:
                # Get active session
                session = instrumentor._active_sessions.pop(id(instance), None)

                # Check if exit was due to an exception
                exc_type = args[0] if args else None

                if session:
                    if exc_type:
                        handler.fail_agent(
                            session,
                            Error(
                                type=exc_type,
                                message=str(args[1]) if len(args) > 1 else "",
                            ),
                        )
                    else:
                        handler.stop_agent(session)

                return await wrapped(*args, **kwargs)
            except Exception as e:
                _LOGGER.debug(f"Error in client exit wrapper: {e}", exc_info=True)
                return await wrapped(*args, **kwargs)

        return traced_exit

    def _client_call_tool_wrapper(self):
        """Wrapper for FastMCP Client.call_tool.

        Uses ToolCall (not Step) to enable MCP-specific metrics:
        - mcp.client.operation.duration
        - mcp.tool.output.size
        """
        handler = self._handler
        instrumentor = self

        async def traced_call_tool(wrapped, instance, args, kwargs):
            import uuid

            # Extract tool name
            tool_name = args[0] if args else kwargs.get("name", "unknown")
            tool_args = args[1] if len(args) > 1 else kwargs.get("arguments", {})

            # Get parent agent invocation for context
            parent_session = instrumentor._active_sessions.get(id(instance))

            # Create a MCPToolCall for proper MCP metrics emission
            tool_call = MCPToolCall(
                name=tool_name,
                arguments=tool_args,
                id=str(uuid.uuid4()),
                framework="fastmcp",
                provider="mcp",
                # Per execute_tool semconv: tool_type indicates type of tool
                # MCP tools are "extension" - executed on agent-side calling external APIs
                tool_type="extension",
                # MCP semantic convention fields for metrics
                mcp_method_name="tools/call",
                network_transport="pipe",  # stdio = pipe
                is_client=True,  # This is client-side
            )

            # Link to parent agent if available
            if parent_session:
                tool_call.agent_name = parent_session.name
                tool_call.agent_id = parent_session.agent_id

            # arguments is already set in constructor above
            # If content capture is enabled and args are complex, serialize them
            if should_capture_content() and tool_args:
                try:
                    serialized = safe_serialize(tool_args)
                    if serialized:
                        tool_call.arguments = truncate_if_needed(serialized)
                except Exception:
                    pass

            handler.start_tool_call(tool_call)

            start_time = time.time()
            try:
                result = await wrapped(*args, **kwargs)

                duration = time.time() - start_time
                tool_call.duration_s = duration

                output_size = 0
                if result:
                    try:
                        serialized = safe_serialize(result)
                        if serialized:
                            output_size = len(serialized.encode("utf-8"))
                            if should_capture_content():
                                tool_call.tool_result = truncate_if_needed(serialized)
                    except Exception:
                        pass

                # Track output size for LLM context awareness
                tool_call.output_size_bytes = output_size

                handler.stop_tool_call(tool_call)
                return result

            except Exception as e:
                duration = time.time() - start_time
                tool_call.duration_s = duration
                tool_call.is_error = True
                handler.fail_tool_call(tool_call, Error(type=type(e), message=str(e)))
                raise

        return traced_call_tool

    def _client_list_tools_wrapper(self):
        """Wrapper for FastMCP Client.list_tools."""
        handler = self._handler

        async def traced_list_tools(wrapped, instance, args, kwargs):
            # Create a Step to represent the list_tools operation
            # Using MCP semantic convention attribute names
            step = Step(
                name="list_tools",
                step_type="admin",
                source="agent",
                framework="fastmcp",
                system="mcp",
            )
            step.attributes["mcp.method.name"] = "tools/list"
            step.attributes["network.transport"] = "pipe"  # stdio = pipe

            handler.start_step(step)

            start_time = time.time()
            try:
                result = await wrapped(*args, **kwargs)

                duration = time.time() - start_time
                step.attributes["mcp.client.operation.duration_s"] = duration

                # Capture tool names (metadata, not content - always captured)
                if result:
                    try:
                        if hasattr(result, "tools"):
                            tool_names = [
                                t.name for t in result.tools if hasattr(t, "name")
                            ]
                            step.attributes["mcp.tools.discovered"] = safe_serialize(
                                tool_names
                            )
                    except Exception:
                        pass

                handler.stop_step(step)
                return result

            except Exception as e:
                duration = time.time() - start_time
                step.attributes["mcp.client.operation.duration_s"] = duration
                step.attributes["error.type"] = type(e).__name__
                handler.fail_step(step, Error(type=type(e), message=str(e)))
                raise

        return traced_list_tools
