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

"""Serialization utilities for IPC between proxy and worker processes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    EvaluationResult,
    GenAI,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Step,
    Text,
    ToolCall,
    Workflow,
)


def serialize_invocation(invocation: GenAI) -> Dict[str, Any]:
    """Convert a GenAI invocation to a serializable dict.

    Extracts only picklable data from the GenAI object, excluding
    Span objects, context tokens, and other non-serializable items.

    Args:
        invocation: The GenAI invocation to serialize.

    Returns:
        A dictionary containing the serializable representation.
    """
    base = {
        "run_id": str(invocation.run_id),
        "parent_run_id": str(invocation.parent_run_id)
        if invocation.parent_run_id
        else None,
        "type": type(invocation).__name__,
        "provider": invocation.provider,
        "framework": invocation.framework,
        "system": invocation.system,
        "agent_name": invocation.agent_name,
        "agent_id": invocation.agent_id,
        "conversation_id": invocation.conversation_id,
        "data_source_id": invocation.data_source_id,
        "sample_for_evaluation": invocation.sample_for_evaluation,
        "evaluation_error": invocation.evaluation_error,
        "start_time": invocation.start_time,
        "end_time": invocation.end_time,
        # Serialize attributes, filtering out non-picklable values
        "attributes": _serialize_attributes(invocation.attributes),
    }

    # Add type-specific fields
    base.update(_extract_type_specific_fields(invocation))

    return base


def _serialize_attributes(attrs: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Serialize attributes dict, filtering non-picklable values."""
    if not attrs:
        return {}
    result = {}
    for key, value in attrs.items():
        try:
            # Test if value is picklable by checking basic types
            if isinstance(
                value, (str, int, float, bool, type(None), list, tuple, dict)
            ):
                result[key] = value
            elif hasattr(value, "__str__"):
                result[key] = str(value)
        except Exception:
            continue
    return result


def _extract_type_specific_fields(invocation: GenAI) -> Dict[str, Any]:
    """Extract type-specific evaluation data."""
    fields: Dict[str, Any] = {}

    if isinstance(invocation, LLMInvocation):
        fields["request_model"] = invocation.request_model
        fields["response_model_name"] = invocation.response_model_name
        fields["operation"] = invocation.operation
        fields["input_tokens"] = invocation.input_tokens
        fields["output_tokens"] = invocation.output_tokens
        fields["input_messages"] = _serialize_input_messages(
            invocation.input_messages
        )
        fields["output_messages"] = _serialize_output_messages(
            invocation.output_messages
        )
        fields["response_finish_reasons"] = list(
            invocation.response_finish_reasons
        )

    elif isinstance(invocation, AgentInvocation):
        fields["name"] = invocation.name
        fields["operation"] = invocation.operation
        fields["agent_type"] = invocation.agent_type
        fields["description"] = invocation.description
        fields["model"] = invocation.model
        fields["tools"] = list(invocation.tools)
        fields["system_instructions"] = invocation.system_instructions
        fields["input_context"] = invocation.input_context
        fields["output_result"] = invocation.output_result

    elif isinstance(invocation, AgentCreation):
        fields["name"] = invocation.name
        fields["operation"] = invocation.operation
        fields["agent_type"] = invocation.agent_type
        fields["description"] = invocation.description
        fields["model"] = invocation.model
        fields["tools"] = list(invocation.tools)
        fields["system_instructions"] = invocation.system_instructions
        fields["input_context"] = invocation.input_context

    elif isinstance(invocation, Workflow):
        fields["name"] = invocation.name
        fields["workflow_type"] = invocation.workflow_type
        fields["description"] = invocation.description
        fields["initial_input"] = invocation.initial_input
        fields["final_output"] = invocation.final_output

    elif isinstance(invocation, Step):
        fields["name"] = invocation.name
        fields["objective"] = invocation.objective
        fields["step_type"] = invocation.step_type
        fields["source"] = invocation.source
        fields["assigned_agent"] = invocation.assigned_agent
        fields["status"] = invocation.status
        fields["description"] = invocation.description
        fields["input_data"] = invocation.input_data
        fields["output_data"] = invocation.output_data

    elif isinstance(invocation, ToolCall):
        fields["name"] = invocation.name
        fields["id"] = invocation.id
        fields["arguments"] = (
            str(invocation.arguments) if invocation.arguments else None
        )

    return fields


def _serialize_input_messages(
    messages: List[InputMessage] | None,
) -> List[Dict[str, Any]]:
    """Serialize input messages to dicts."""
    if not messages:
        return []
    result = []
    for msg in messages:
        result.append(
            {
                "role": msg.role,
                "parts": _serialize_message_parts(msg.parts),
            }
        )
    return result


def _serialize_output_messages(
    messages: List[OutputMessage] | None,
) -> List[Dict[str, Any]]:
    """Serialize output messages to dicts."""
    if not messages:
        return []
    result = []
    for msg in messages:
        result.append(
            {
                "role": msg.role,
                "parts": _serialize_message_parts(msg.parts),
                "finish_reason": msg.finish_reason,
            }
        )
    return result


def _serialize_message_parts(parts: List[Any] | None) -> List[Dict[str, Any]]:
    """Serialize message parts to dicts."""
    if not parts:
        return []
    result = []
    for part in parts:
        if isinstance(part, Text):
            result.append({"type": "text", "content": part.content})
        elif isinstance(part, ToolCall):
            result.append(
                {
                    "type": "tool_call",
                    "name": part.name,
                    "id": part.id,
                    "arguments": str(part.arguments)
                    if part.arguments
                    else None,
                }
            )
        elif isinstance(part, str):
            result.append({"type": "text", "content": part})
        else:
            # Fallback for other types
            result.append({"type": "unknown", "content": str(part)})
    return result


def deserialize_invocation(payload: Dict[str, Any]) -> GenAI:
    """Reconstruct a GenAI invocation from serialized data.

    Args:
        payload: The serialized dictionary from serialize_invocation.

    Returns:
        A reconstructed GenAI invocation object.
    """
    from uuid import UUID

    inv_type = payload.get("type", "")

    # Common fields
    common_kwargs: Dict[str, Any] = {
        "provider": payload.get("provider"),
        "framework": payload.get("framework"),
        "system": payload.get("system"),
        "agent_name": payload.get("agent_name"),
        "agent_id": payload.get("agent_id"),
        "conversation_id": payload.get("conversation_id"),
        "data_source_id": payload.get("data_source_id"),
        "sample_for_evaluation": payload.get("sample_for_evaluation", True),
        "evaluation_error": payload.get("evaluation_error"),
        "start_time": payload.get("start_time", 0.0),
        "end_time": payload.get("end_time"),
        "attributes": payload.get("attributes", {}),
    }

    # Parse run_id if present
    run_id_str = payload.get("run_id")
    if run_id_str:
        try:
            common_kwargs["run_id"] = UUID(run_id_str)
        except ValueError:
            pass

    parent_run_id_str = payload.get("parent_run_id")
    if parent_run_id_str:
        try:
            common_kwargs["parent_run_id"] = UUID(parent_run_id_str)
        except ValueError:
            pass

    if inv_type == "LLMInvocation":
        return LLMInvocation(
            request_model=payload.get("request_model", ""),
            response_model_name=payload.get("response_model_name"),
            operation=payload.get("operation", "chat"),
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
            input_messages=_deserialize_input_messages(
                payload.get("input_messages", [])
            ),
            output_messages=_deserialize_output_messages(
                payload.get("output_messages", [])
            ),
            response_finish_reasons=payload.get("response_finish_reasons", []),
            **common_kwargs,
        )

    elif inv_type == "AgentInvocation":
        return AgentInvocation(
            name=payload.get("name", ""),
            agent_type=payload.get("agent_type"),
            description=payload.get("description"),
            model=payload.get("model"),
            tools=payload.get("tools", []),
            system_instructions=payload.get("system_instructions"),
            input_context=payload.get("input_context"),
            output_result=payload.get("output_result"),
            **common_kwargs,
        )

    elif inv_type == "AgentCreation":
        return AgentCreation(
            name=payload.get("name", ""),
            agent_type=payload.get("agent_type"),
            description=payload.get("description"),
            model=payload.get("model"),
            tools=payload.get("tools", []),
            system_instructions=payload.get("system_instructions"),
            input_context=payload.get("input_context"),
            **common_kwargs,
        )

    elif inv_type == "Workflow":
        return Workflow(
            name=payload.get("name", ""),
            workflow_type=payload.get("workflow_type"),
            description=payload.get("description"),
            initial_input=payload.get("initial_input"),
            final_output=payload.get("final_output"),
            **common_kwargs,
        )

    elif inv_type == "Step":
        return Step(
            name=payload.get("name", ""),
            objective=payload.get("objective"),
            step_type=payload.get("step_type"),
            source=payload.get("source"),
            assigned_agent=payload.get("assigned_agent"),
            status=payload.get("status"),
            description=payload.get("description"),
            input_data=payload.get("input_data"),
            output_data=payload.get("output_data"),
            **common_kwargs,
        )

    elif inv_type == "ToolCall":
        return ToolCall(
            name=payload.get("name", ""),
            id=payload.get("id"),
            arguments=payload.get("arguments"),
            **common_kwargs,
        )

    # Fallback to base GenAI - this shouldn't happen in practice
    # but provides safety
    raise ValueError(f"Unknown invocation type: {inv_type}")


def _deserialize_input_messages(
    messages: List[Dict[str, Any]],
) -> List[InputMessage]:
    """Deserialize input messages from dicts."""
    result = []
    for msg_dict in messages:
        parts = _deserialize_message_parts(msg_dict.get("parts", []))
        result.append(InputMessage(role=msg_dict.get("role", ""), parts=parts))
    return result


def _deserialize_output_messages(
    messages: List[Dict[str, Any]],
) -> List[OutputMessage]:
    """Deserialize output messages from dicts."""
    result = []
    for msg_dict in messages:
        parts = _deserialize_message_parts(msg_dict.get("parts", []))
        result.append(
            OutputMessage(
                role=msg_dict.get("role", ""),
                parts=parts,
                finish_reason=msg_dict.get("finish_reason", "stop"),
            )
        )
    return result


def _deserialize_message_parts(parts: List[Dict[str, Any]]) -> List[Any]:
    """Deserialize message parts from dicts."""
    result = []
    for part_dict in parts:
        part_type = part_dict.get("type", "text")
        if part_type == "text":
            result.append(Text(content=part_dict.get("content", "")))
        elif part_type == "tool_call":
            result.append(
                ToolCall(
                    name=part_dict.get("name", ""),
                    id=part_dict.get("id"),
                    arguments=part_dict.get("arguments"),
                )
            )
        else:
            # Fallback to text
            result.append(
                Text(content=part_dict.get("content", str(part_dict)))
            )
    return result


def serialize_evaluation_result(result: EvaluationResult) -> Dict[str, Any]:
    """Serialize an EvaluationResult to a dict."""
    return {
        "metric_name": result.metric_name,
        "score": result.score,
        "label": result.label,
        "explanation": result.explanation,
        "error": {
            "message": result.error.message,
            "type": result.error.type.__name__,
        }
        if result.error
        else None,
        "attributes": dict(result.attributes) if result.attributes else {},
    }


def deserialize_evaluation_result(data: Dict[str, Any]) -> EvaluationResult:
    """Deserialize an EvaluationResult from a dict."""
    from opentelemetry.util.genai.types import Error

    error = None
    if data.get("error"):
        error_data = data["error"]
        # We can't reconstruct the exact exception type, use generic Exception
        error = Error(message=error_data.get("message", ""), type=Exception)

    return EvaluationResult(
        metric_name=data.get("metric_name", ""),
        score=data.get("score"),
        label=data.get("label"),
        explanation=data.get("explanation"),
        error=error,
        attributes=data.get("attributes", {}),
    )


__all__ = [
    "serialize_invocation",
    "deserialize_invocation",
    "serialize_evaluation_result",
    "deserialize_evaluation_result",
]
