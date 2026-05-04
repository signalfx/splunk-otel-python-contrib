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

from .utils import bind_call_arguments, safe_json_dumps, safe_str, truncate_error

_LOGGER = logging.getLogger(__name__)


def _record_tool_call_error(
    handler: TelemetryHandler, tool_call: ToolCall, error: Exception
) -> None:
    try:
        handler.fail_tool_call(
            tool_call, Error(type=type(error), message=truncate_error(error))
        )
    except Exception:
        _LOGGER.debug("Failed to record memory tool call error.", exc_info=True)


def _record_retrieval_error(
    handler: TelemetryHandler, invocation: RetrievalInvocation, error: Exception
) -> None:
    try:
        handler.fail_retrieval(
            invocation, Error(type=type(error), message=truncate_error(error))
        )
    except Exception:
        _LOGGER.debug("Failed to record memory retrieval error.", exc_info=True)


def _finish_tool_call(handler: TelemetryHandler, tool_call: ToolCall) -> None:
    try:
        handler.stop_tool_call(tool_call)
    except Exception:
        _LOGGER.debug("Failed to finish memory tool call.", exc_info=True)


def _finish_retrieval(
    handler: TelemetryHandler, invocation: RetrievalInvocation
) -> None:
    try:
        handler.stop_retrieval(invocation)
    except Exception:
        _LOGGER.debug("Failed to finish memory retrieval.", exc_info=True)


def wrap_memory_retrieve(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap MemoryClient.retrieve_memories to create a RetrievalInvocation span.

    Args:
        wrapped: Original retrieve_memories method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture query/result content in spans

    Returns:
        Result of original retrieve_memories
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        query = call_arguments.get("query", "")
        top_k = call_arguments.get("top_k", 3)
        invocation = RetrievalInvocation(
            retriever_type="bedrock-agentcore-memory",
            query=safe_str(query) if capture_content else "",
            top_k=top_k,
        )
        handler.start_retrieval(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        _record_retrieval_error(handler, invocation, e)
        raise

    try:
        if isinstance(result, (list, dict)):
            records = (
                result if isinstance(result, list) else result.get("memoryRecords", [])
            )
            invocation.documents_retrieved = len(records)
    except Exception:
        _LOGGER.debug("Failed to enrich memory retrieval.", exc_info=True)

    _finish_retrieval(handler, invocation)
    return result


def wrap_memory_create_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap MemoryClient.create_event to create a ToolCall span.

    Args:
        wrapped: Original create_event method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture arguments/result content in spans

    Returns:
        Result of original create_event
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        memory_id = call_arguments.get("memory_id")
        actor_id = call_arguments.get("actor_id")
        session_id = call_arguments.get("session_id")
        invocation = ToolCall(
            name="memory.create_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(memory_id),
                    "actor_id": safe_str(actor_id),
                    "session_id": safe_str(session_id),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        _record_tool_call_error(handler, invocation, e)
        raise

    try:
        if capture_content and result is not None:
            invocation.tool_result = (
                safe_json_dumps(result) if not isinstance(result, str) else result
            )
    except Exception:
        _LOGGER.debug("Failed to enrich memory create_event tool call.", exc_info=True)

    _finish_tool_call(handler, invocation)
    return result


def wrap_memory_create_blob_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap MemoryClient.create_blob_event to create a ToolCall span.

    Args:
        wrapped: Original create_blob_event method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture arguments/result content in spans

    Returns:
        Result of original create_blob_event
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        memory_id = call_arguments.get("memory_id")
        actor_id = call_arguments.get("actor_id")
        session_id = call_arguments.get("session_id")
        invocation = ToolCall(
            name="memory.create_blob_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(memory_id),
                    "actor_id": safe_str(actor_id),
                    "session_id": safe_str(session_id),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        _record_tool_call_error(handler, invocation, e)
        raise

    try:
        if capture_content and result is not None:
            invocation.tool_result = (
                safe_json_dumps(result) if not isinstance(result, str) else result
            )
    except Exception:
        _LOGGER.debug(
            "Failed to enrich memory create_blob_event tool call.", exc_info=True
        )

    _finish_tool_call(handler, invocation)
    return result


def wrap_memory_list_events(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    """Wrap MemoryClient.list_events to create a ToolCall span.

    Args:
        wrapped: Original list_events method
        instance: MemoryClient instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: TelemetryHandler instance
        capture_content: Whether to capture arguments/result content in spans

    Returns:
        Result of original list_events
    """
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        memory_id = call_arguments.get("memory_id")
        invocation = ToolCall(
            name="memory.list_events",
            arguments=safe_json_dumps({"memory_id": safe_str(memory_id)})
            if capture_content
            else None,
            system="bedrock-agentcore",
        )

        handler.start_tool_call(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        _record_tool_call_error(handler, invocation, e)
        raise

    try:
        if capture_content and result is not None:
            invocation.tool_result = (
                safe_json_dumps(result) if not isinstance(result, str) else result
            )
    except Exception:
        _LOGGER.debug("Failed to enrich memory list_events tool call.", exc_info=True)

    _finish_tool_call(handler, invocation)
    return result


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
        capture_content: bool = False,
    ) -> Any:
        try:
            invocation = ToolCall(
                name=f"memory.{operation_name}",
                arguments=safe_json_dumps(kwargs)
                if capture_content and kwargs
                else None,
                system="bedrock-agentcore",
            )
            handler.start_tool_call(invocation)
        except Exception:
            return wrapped(*args, **kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:
            _record_tool_call_error(handler, invocation, e)
            raise

        try:
            if capture_content and result is not None:
                invocation.tool_result = (
                    safe_json_dumps(result) if not isinstance(result, str) else result
                )
        except Exception:
            _LOGGER.debug(
                "Failed to enrich memory %s tool call.",
                operation_name,
                exc_info=True,
            )

        _finish_tool_call(handler, invocation)
        return result

    return wrapper
