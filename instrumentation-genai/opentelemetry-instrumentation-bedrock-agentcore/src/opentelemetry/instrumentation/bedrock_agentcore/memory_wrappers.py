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

import inspect
import json
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error, RetrievalInvocation, ToolCall

from .utils import (
    bind_call_arguments,
    invoke_tool_call,
    safe_json_dumps,
    safe_str,
    truncate_error,
)


def wrap_memory_retrieve(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        query = call_arguments.get("query", "")
        top_k = call_arguments.get("top_k", 3)
        invocation = RetrievalInvocation(
            operation_name="retrieval",
            provider="bedrock-agentcore-memory",
            retriever_type="bedrock-agentcore-memory",
            data_source_id="memory.retrieve_memories",
            query=safe_str(query) if capture_content else "",
            top_k=top_k,
            system="bedrock-agentcore",
        )
        handler.start_retrieval(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_retrieval(
            invocation, Error(type=type(e), message=truncate_error(e))
        )
        raise

    if isinstance(result, (list, dict)):
        records = (
            result if isinstance(result, list) else result.get("memoryRecords", [])
        )
        invocation.documents_retrieved = len(records)

    handler.stop_retrieval(invocation)
    return result


def wrap_memory_create_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        invocation = ToolCall(
            name="memory.create_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(call_arguments.get("memory_id")),
                    "actor_id": safe_str(call_arguments.get("actor_id")),
                    "session_id": safe_str(call_arguments.get("session_id")),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — event responses can include message payloads
    return invoke_tool_call(
        handler, invocation, wrapped, args, kwargs, capture_content=False
    )


def wrap_memory_create_blob_event(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        invocation = ToolCall(
            name="memory.create_blob_event",
            arguments=safe_json_dumps(
                {
                    "memory_id": safe_str(call_arguments.get("memory_id")),
                    "actor_id": safe_str(call_arguments.get("actor_id")),
                    "session_id": safe_str(call_arguments.get("session_id")),
                }
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — blob event responses can include uploaded content
    return invoke_tool_call(
        handler, invocation, wrapped, args, kwargs, capture_content=False
    )


def wrap_memory_list_events(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        invocation = ToolCall(
            name="memory.list_events",
            arguments=safe_json_dumps(
                {"memory_id": safe_str(call_arguments.get("memory_id"))}
            )
            if capture_content
            else None,
            system="bedrock-agentcore",
        )
    except Exception:
        return wrapped(*args, **kwargs)

    # never capture tool_result — list_events can include event payloads by default
    return invoke_tool_call(
        handler, invocation, wrapped, args, kwargs, capture_content=False
    )


_CONVERSATION_SAFE_ARGUMENTS = frozenset(
    {
        "actor_id",
        "branch_id",
        "branch_name",
        "conversation_id",
        "k",
        "last_k",
        "limit",
        "max_results",
        "max_turns",
        "memory_id",
        "namespace",
        "session_id",
        "source_branch_id",
        "target_branch_id",
    }
)


def _safe_conversation_arguments(call_arguments: dict[str, Any]) -> dict[str, Any]:
    safe_args = {}
    for key in _CONVERSATION_SAFE_ARGUMENTS:
        value = call_arguments.get(key)
        if value is None or value == "" or callable(value):
            continue
        if isinstance(value, (bool, int, float)):
            safe_args[key] = value
        else:
            safe_args[key] = safe_str(value)
    return safe_args


def wrap_memory_conversation_operation(operation_name: str) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
            safe_args = _safe_conversation_arguments(call_arguments)
            invocation = ToolCall(
                name=f"memory.{operation_name}",
                arguments=json.dumps(safe_args, default=str)
                if capture_content and safe_args
                else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        # never capture tool_result — conversation responses can include messages
        return invoke_tool_call(
            handler, invocation, wrapped, args, kwargs, capture_content=False
        )

    return wrapper


_SESSION_SAFE_ARGUMENTS = frozenset(
    {
        "actor_id",
        "branch_name",
        "event_id",
        "include_parent_branches",
        "include_payload",
        "k",
        "max_results",
        "memory_id",
        "namespace",
        "namespace_prefix",
        "record_id",
        "root_event_id",
        "session_id",
        "strategy_id",
    }
)


def _safe_session_arguments(
    call_arguments: dict[str, Any], instance: Any
) -> dict[str, Any]:
    safe_args = {}
    memory_id = getattr(instance, "_memory_id", None) or getattr(
        instance, "memory_id", None
    )
    if memory_id is not None and memory_id != "":
        safe_args["memory_id"] = safe_str(memory_id)

    for key in _SESSION_SAFE_ARGUMENTS:
        value = call_arguments.get(key)
        if value is None or value == "" or callable(value):
            continue
        if isinstance(value, (bool, int, float)):
            safe_args[key] = value
        else:
            safe_args[key] = safe_str(value)
    return safe_args


def wrap_memory_session_operation(operation_name: str) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
            safe_args = _safe_session_arguments(call_arguments, instance)
            invocation = ToolCall(
                name=f"memory.session.{operation_name}",
                arguments=json.dumps(safe_args, default=str)
                if capture_content and safe_args
                else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        # never capture tool_result - session APIs can return events or records
        return invoke_tool_call(
            handler, invocation, wrapped, args, kwargs, capture_content=False
        )

    return wrapper


def wrap_memory_session_async_operation(operation_name: str) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
            safe_args = _safe_session_arguments(call_arguments, instance)
            invocation = ToolCall(
                name=f"memory.session.{operation_name}",
                arguments=json.dumps(safe_args, default=str)
                if capture_content and safe_args
                else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        async def _invoke() -> Any:
            handler.start_tool_call(invocation)
            try:
                result = wrapped(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                handler.fail_tool_call(
                    invocation, Error(type=type(e), message=truncate_error(e))
                )
                raise

            handler.stop_tool_call(invocation)
            return result

        return _invoke()

    return wrapper


def wrap_memory_session_search_long_term_memories(
    wrapped: Any,
    instance: Any,
    args: tuple,
    kwargs: dict,
    handler: TelemetryHandler,
    capture_content: bool = False,
) -> Any:
    try:
        call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
        query = call_arguments.get("query", "")
        top_k = call_arguments.get("top_k", 3)
        invocation = RetrievalInvocation(
            operation_name="retrieval",
            provider="bedrock-agentcore-memory",
            retriever_type="bedrock-agentcore-memory",
            data_source_id="memory.session.search_long_term_memories",
            query=safe_str(query) if capture_content else "",
            top_k=top_k,
            system="bedrock-agentcore",
        )
        handler.start_retrieval(invocation)
    except Exception:
        return wrapped(*args, **kwargs)

    try:
        result = wrapped(*args, **kwargs)
    except Exception as e:
        handler.fail_retrieval(
            invocation, Error(type=type(e), message=truncate_error(e))
        )
        raise

    if isinstance(result, (list, dict)):
        records = (
            result
            if isinstance(result, list)
            else result.get("memoryRecords")
            or result.get("memoryRecordSummaries")
            or []
        )
        invocation.documents_retrieved = len(records)

    handler.stop_retrieval(invocation)
    return result


_MEMORY_OPERATION_SAFE_ARGUMENTS = frozenset(
    {
        "memory_id",
        "memoryId",
        "memory_name",
        "memoryName",
        "name",
        "max_results",
        "maxResults",
        "status",
        "strategy_id",
        "strategyId",
        "strategy_name",
        "strategyName",
        "strategy_type",
        "strategyType",
    }
)


def _safe_memory_operation_arguments(call_arguments: dict[str, Any]) -> dict[str, Any]:
    safe_args = {}
    for key in _MEMORY_OPERATION_SAFE_ARGUMENTS:
        value = call_arguments.get(key)
        if value is None or value == "" or callable(value):
            continue
        if isinstance(value, (bool, int, float)):
            safe_args[key] = value
        else:
            safe_args[key] = safe_str(value)
    return safe_args


def wrap_memory_operation(operation_name: str) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple,
        kwargs: dict,
        handler: TelemetryHandler,
        capture_content: bool = False,
    ) -> Any:
        try:
            call_arguments = bind_call_arguments(wrapped, instance, args, kwargs)
            safe_args = _safe_memory_operation_arguments(call_arguments)
            invocation = ToolCall(
                name=f"memory.{operation_name}",
                arguments=safe_json_dumps(safe_args)
                if capture_content and safe_args
                else None,
                system="bedrock-agentcore",
            )
        except Exception:
            return wrapped(*args, **kwargs)

        # never capture tool_result - MemoryClient generic operations can include
        # IAM roles, stream delivery resources, and strategy configuration.
        return invoke_tool_call(
            handler, invocation, wrapped, args, kwargs, capture_content=False
        )

    return wrapper
