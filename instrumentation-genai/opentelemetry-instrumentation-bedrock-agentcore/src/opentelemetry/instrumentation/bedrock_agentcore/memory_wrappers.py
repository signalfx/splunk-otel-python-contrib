# Copyright Splunk Inc.
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

"""Wrapt wrappers for Bedrock AgentCore Memory instrumentation."""

import logging
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, RetrievalInvocation, ToolCall

from .utils import safe_json_dumps, safe_str

_LOGGER = logging.getLogger(__name__)


def wrap_memory_retrieve(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap MemoryClient.retrieve_memories to create a RetrievalInvocation span.

    Args:
        wrapped: Original retrieve_memories method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original retrieve_memories
    """
    try:
        # Signature: retrieve_memories(memory_id, namespace, query, actor_id=None, top_k=3)
        query = args[2] if len(args) > 2 else kwargs.get("query", "")
        top_k = args[4] if len(args) > 4 else kwargs.get("top_k")

        invocation = RetrievalInvocation(
            retriever_type="bedrock-agentcore-memory",
            query=safe_str(query),
            top_k=top_k,
        )

        handler.start_retrieval(invocation)

        try:
            result = wrapped(*args, **kwargs)

            # Count retrieved records
            if result and isinstance(result, (list, dict)):
                records = (
                    result
                    if isinstance(result, list)
                    else result.get("memoryRecords", [])
                )
                invocation.documents_retrieved = len(records)

            handler.stop_retrieval(invocation)

            return result
        except Exception as e:
            handler.fail_retrieval(
                invocation, Error(type=type(e).__name__, message=safe_str(e))
            )
            raise
    except Exception:
        return wrapped(*args, **kwargs)


def wrap_memory_create_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap MemoryClient.create_event to create a ToolCall span.

    Args:
        wrapped: Original create_event method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original create_event
    """
    try:
        memory_id = kwargs.get("memory_id") or (args[0] if args else None)
        actor_id = kwargs.get("actor_id") or (args[1] if len(args) > 1 else None)
        session_id = kwargs.get("session_id") or (args[2] if len(args) > 2 else None)

        invocation = ToolCall(
            name="memory.create_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(memory_id),
                    "actor_id": safe_str(actor_id),
                    "session_id": safe_str(session_id),
                }
            ),
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)

        try:
            result = wrapped(*args, **kwargs)

            if result is not None:
                invocation.tool_result = (
                    safe_json_dumps(result) if not isinstance(result, str) else result
                )

            handler.stop_tool_call(invocation)

            return result
        except Exception as e:
            handler.fail_tool_call(
                invocation, Error(type=type(e).__name__, message=safe_str(e))
            )
            raise
    except Exception:
        return wrapped(*args, **kwargs)


def wrap_memory_create_blob_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap MemoryClient.create_blob_event to create a ToolCall span.

    Args:
        wrapped: Original create_blob_event method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original create_blob_event
    """
    try:
        memory_id = args[0] if len(args) > 0 else kwargs.get("memory_id")
        actor_id = args[1] if len(args) > 1 else kwargs.get("actor_id")
        session_id = args[2] if len(args) > 2 else kwargs.get("session_id")

        invocation = ToolCall(
            name="memory.create_blob_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(memory_id),
                    "actor_id": safe_str(actor_id),
                    "session_id": safe_str(session_id),
                }
            ),
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)

        try:
            result = wrapped(*args, **kwargs)

            if result is not None:
                invocation.tool_result = (
                    safe_json_dumps(result) if not isinstance(result, str) else result
                )

            handler.stop_tool_call(invocation)

            return result
        except Exception as e:
            handler.fail_tool_call(
                invocation, Error(type=type(e).__name__, message=safe_str(e))
            )
            raise
    except Exception:
        return wrapped(*args, **kwargs)


def wrap_memory_list_events(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
) -> Any:
    """Wrap MemoryClient.list_events to create a ToolCall span.

    Args:
        wrapped: Original list_events method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance

    Returns:
        Result of original list_events
    """
    try:
        memory_id = kwargs.get("memory_id") or (args[0] if args else None)

        invocation = ToolCall(
            name="memory.list_events",
            arguments=safe_json_dumps({"memory_id": safe_str(memory_id)}),
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)

        try:
            result = wrapped(*args, **kwargs)

            if result is not None:
                invocation.tool_result = (
                    safe_json_dumps(result) if not isinstance(result, str) else result
                )

            handler.stop_tool_call(invocation)

            return result
        except Exception as e:
            handler.fail_tool_call(
                invocation, Error(type=type(e).__name__, message=safe_str(e))
            )
            raise
    except Exception:
        return wrapped(*args, **kwargs)


def wrap_memory_operation(
    operation_name: str,
) -> Any:
    """Generic wrapper factory for MemoryClient operations that creates ToolCall spans.

    Args:
        operation_name: Name of the operation (e.g., "create_memory", "delete_memory")

    Returns:
        Wrapper function
    """
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
    ) -> Any:
        try:
            invocation = ToolCall(
                name=f"memory.{operation_name}",
                arguments=safe_json_dumps(kwargs) if kwargs else safe_json_dumps({}),
                system="bedrock-agentcore",
            )

            handler.start_tool_call(invocation)

            try:
                result = wrapped(*args, **kwargs)

                if result is not None:
                    invocation.tool_result = (
                        safe_json_dumps(result) if not isinstance(result, str) else result
                    )

                handler.stop_tool_call(invocation)

                return result
            except Exception as e:
                handler.fail_tool_call(
                    invocation, Error(type=type(e).__name__, message=safe_str(e))
                )
                raise
        except Exception:
            return wrapped(*args, **kwargs)

    return wrapper
