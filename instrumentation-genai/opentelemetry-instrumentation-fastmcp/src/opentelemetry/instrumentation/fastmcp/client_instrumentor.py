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
from typing import Any, Callable

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    MCPOperation,
    MCPToolCall,
)
from opentelemetry.instrumentation.fastmcp.utils import (
    detect_transport,
    safe_serialize,
    should_capture_content,
    truncate_if_needed,
)

_LOGGER = logging.getLogger(__name__)


def _traced_mcp_operation(
    handler: TelemetryHandler,
    build_op: Callable[[Any, str], MCPOperation],
) -> Callable:
    """Generic async wrapper for MCPOperation lifecycle.

    ``build_op(instance, transport)`` must return a ready-to-start
    :class:`MCPOperation`. The wrapper handles start / stop / fail
    and duration timing.
    """

    async def wrapper(wrapped, instance, args, kwargs):
        transport = detect_transport(instance)
        op = build_op(instance, transport)

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

    return wrapper


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
        hooks = {
            "Client.__aenter__": self._client_enter_wrapper(),
            "Client.__aexit__": self._client_exit_wrapper(),
            "Client.call_tool": self._client_call_tool_wrapper(),
            "Client.list_tools": self._client_list_tools_wrapper(),
            "Client.read_resource": self._client_read_resource_wrapper(),
            "Client.get_prompt": self._client_get_prompt_wrapper(),
        }
        for target, wrapper in hooks.items():
            register_post_import_hook(
                lambda _, t=target, w=wrapper: wrap_function_wrapper(
                    "fastmcp.client", t, w
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
                session.attributes["network.transport"] = "pipe"  # stdio = pipe

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
                        session.attributes["error.type"] = (
                            exc_type.__qualname__
                            if isinstance(exc_type, type)
                            else str(exc_type)
                        )
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
    # tools/call (unique logic — not DRY-able with MCPOperation helpers)
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
            transport = detect_transport(instance)

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
    # MCPOperation wrappers (list, read, get)
    # ------------------------------------------------------------------
    def _client_list_tools_wrapper(self):
        """Wrapper for FastMCP Client.list_tools."""
        return _traced_mcp_operation(
            self._handler,
            lambda _inst, transport: MCPOperation(
                target="",
                mcp_method_name="tools/list",
                network_transport=transport,
                is_client=True,
                framework="fastmcp",
                system="mcp",
            ),
        )

    def _client_read_resource_wrapper(self):
        """Wrapper for FastMCP Client.read_resource."""
        handler = self._handler

        async def traced_read_resource(wrapped, instance, args, kwargs):
            uri = str(args[0]) if args else str(kwargs.get("uri", ""))
            transport = detect_transport(instance)
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

    def _client_get_prompt_wrapper(self):
        """Wrapper for FastMCP Client.get_prompt."""
        handler = self._handler

        async def traced_get_prompt(wrapped, instance, args, kwargs):
            prompt_name = str(args[0]) if args else str(kwargs.get("name", ""))
            transport = detect_transport(instance)
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
