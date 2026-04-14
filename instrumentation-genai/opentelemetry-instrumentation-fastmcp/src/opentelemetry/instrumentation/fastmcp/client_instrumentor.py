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
    Error,
    MCPOperation,
    MCPToolCall,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    safe_serialize,
    should_capture_content,
    truncate_if_needed,
)

_LOGGER = logging.getLogger(__name__)


def _detect_client_transport(client_instance: object) -> str:
    """Best-effort transport detection from a FastMCP Client instance."""
    try:
        transport = getattr(client_instance, "transport", None)
        if transport is not None:
            cls_name = type(transport).__name__.lower()
            if "sse" in cls_name or "streamable" in cls_name:
                return "tcp"
    except Exception:
        pass
    return "pipe"


class ClientInstrumentor:
    """Handles FastMCP client-side instrumentation.

    Instruments:
    - FastMCP Client.__aenter__: Start client session trace
    - FastMCP Client.__aexit__: End client session trace
    - Client.call_tool, list_tools, read_resource, get_prompt
    """

    def __init__(self, telemetry_handler: TelemetryHandler):
        self._handler = telemetry_handler
        self._active_sessions: dict[int, AgentInvocation] = {}

    def instrument(self):
        """Apply FastMCP client-side instrumentation."""
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

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.read_resource",
                self._client_read_resource_wrapper(),
            ),
            "fastmcp.client",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.get_prompt",
                self._client_get_prompt_wrapper(),
            ),
            "fastmcp.client",
        )

    def uninstrument(self):
        """Remove FastMCP client-side instrumentation.

        Note: wrapt doesn't provide a clean way to unwrap post-import hooks.
        """
        pass

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _client_enter_wrapper(self):
        """Wrapper for FastMCP Client.__aenter__ to start a session trace."""
        instrumentor = self
        handler = self._handler

        async def traced_enter(wrapped, instance, args, kwargs):
            try:
                result = await wrapped(*args, **kwargs)

                session = AgentInvocation(
                    name="mcp.client",
                    agent_type="mcp_client",
                    framework="fastmcp",
                    system="mcp",
                )
                session.attributes["gen_ai.operation.name"] = "mcp.client_session"

                instrumentor._active_sessions[id(instance)] = session
                handler.start_agent(session)
                return result
            except Exception as e:
                _LOGGER.debug("Error in client enter wrapper: %s", e, exc_info=True)
                return await wrapped(*args, **kwargs)

        return traced_enter

    def _client_exit_wrapper(self):
        """Wrapper for FastMCP Client.__aexit__ to end the session trace."""
        instrumentor = self
        handler = self._handler

        async def traced_exit(wrapped, instance, args, kwargs):
            try:
                session = instrumentor._active_sessions.pop(id(instance), None)
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
                _LOGGER.debug("Error in client exit wrapper: %s", e, exc_info=True)
                return await wrapped(*args, **kwargs)

        return traced_exit

    # ------------------------------------------------------------------
    # tools/call
    # ------------------------------------------------------------------
    def _client_call_tool_wrapper(self):
        """Wrapper for FastMCP Client.call_tool."""
        handler = self._handler
        instrumentor = self

        async def traced_call_tool(wrapped, instance, args, kwargs):
            import uuid

            tool_name = args[0] if args else kwargs.get("name", "unknown")
            tool_args = args[1] if len(args) > 1 else kwargs.get("arguments", {})

            parent_session = instrumentor._active_sessions.get(id(instance))
            transport = _detect_client_transport(instance)

            tool_call = MCPToolCall(
                name=tool_name,
                arguments=tool_args,
                id=str(uuid.uuid4()),
                framework="fastmcp",
                provider="mcp",
                tool_type="extension",
                mcp_method_name="tools/call",
                network_transport=transport,
                is_client=True,
            )

            if parent_session:
                tool_call.agent_name = parent_session.name
                tool_call.agent_id = parent_session.agent_id

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

    # ------------------------------------------------------------------
    # tools/list  (MCPOperation instead of Step)
    # ------------------------------------------------------------------
    def _client_list_tools_wrapper(self):
        """Wrapper for FastMCP Client.list_tools."""
        handler = self._handler

        async def traced_list_tools(wrapped, instance, args, kwargs):
            transport = _detect_client_transport(instance)
            op = MCPOperation(
                target="",
                mcp_method_name="tools/list",
                network_transport=transport,
                is_client=True,
                framework="fastmcp",
                system="mcp",
            )

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

        return traced_list_tools

    # ------------------------------------------------------------------
    # resources/read
    # ------------------------------------------------------------------
    def _client_read_resource_wrapper(self):
        """Wrapper for FastMCP Client.read_resource."""
        handler = self._handler

        async def traced_read_resource(wrapped, instance, args, kwargs):
            uri = ""
            if args:
                uri = str(args[0])
            elif "uri" in kwargs:
                uri = str(kwargs["uri"])

            transport = _detect_client_transport(instance)
            op = MCPOperation(
                target=uri,
                mcp_method_name="resources/read",
                network_transport=transport,
                mcp_resource_uri=uri or None,
                is_client=True,
                framework="fastmcp",
                system="mcp",
            )

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
    def _client_get_prompt_wrapper(self):
        """Wrapper for FastMCP Client.get_prompt."""
        handler = self._handler

        async def traced_get_prompt(wrapped, instance, args, kwargs):
            prompt_name = ""
            if args:
                prompt_name = str(args[0])
            elif "name" in kwargs:
                prompt_name = str(kwargs["name"])

            transport = _detect_client_transport(instance)
            op = MCPOperation(
                target=prompt_name,
                mcp_method_name="prompts/get",
                network_transport=transport,
                gen_ai_prompt_name=prompt_name or None,
                is_client=True,
                framework="fastmcp",
                system="mcp",
            )

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
