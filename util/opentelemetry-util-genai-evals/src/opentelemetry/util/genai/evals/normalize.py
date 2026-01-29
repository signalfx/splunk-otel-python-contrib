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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    GenAI,
    LLMInvocation,
    Text,
    ToolCall,
)


@dataclass(frozen=True)
class CanonicalEvalCase:
    type_name: str
    input_text: str
    output_text: str
    context: list[str] | None
    retrieval_context: list[str] | None
    metadata: dict[str, Any]
    is_tool_only_llm: bool
    is_agent_non_invoke: bool


def _dedupe_preserve(seq: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in seq:
        if item and item not in seen:
            seen[item] = None
    return list(seen.keys())


def flatten_to_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("content", "page_content", "text", "body", "value"):
            try:
                inner = value.get(key)  # type: ignore[index]
            except Exception:
                inner = None
            if isinstance(inner, str):
                return [inner]
            if inner is not None:
                return flatten_to_strings(inner)
        try:
            coerced = str(value)
            return [coerced]
        except Exception:
            return []
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        flattened: list[str] = []
        for item in value:
            flattened.extend(flatten_to_strings(item))
        return flattened
    return [str(value)]


def extract_text_from_messages(messages: Sequence[Any]) -> str:
    chunks: list[str] = []
    for message in messages or []:
        parts = getattr(message, "parts", [])
        for part in parts:
            if isinstance(part, Text):
                if part.content:
                    chunks.append(part.content)
    return "\n".join(c for c in chunks if c).strip()


def extract_context(attributes: Mapping[str, Any] | None) -> list[str] | None:
    if not isinstance(attributes, Mapping):
        return None
    context_values: list[str] = []
    for key in ("context", "additional_context"):
        context_values.extend(flatten_to_strings(attributes.get(key)))
    context_values = [v for v in context_values if v]
    return context_values or None


def extract_retrieval_context(
    attributes: Mapping[str, Any] | None,
) -> list[str] | None:
    if not isinstance(attributes, Mapping):
        return None
    retrieval_values: list[str] = []
    for key in (
        "retrieval_context",
        "retrieved_context",
        "retrieved_documents",
        "documents",
        "sources",
        "evidence",
    ):
        retrieval_values.extend(flatten_to_strings(attributes.get(key)))
    retrieval_values = [v for v in retrieval_values if v]
    return retrieval_values or None


def _collect_agent_inputs(agent: AgentInvocation) -> list[str]:
    chunks: list[str] = []
    if agent.system_instructions:
        chunks.extend(flatten_to_strings(agent.system_instructions))
    # Extract text from structured input_messages
    for msg in getattr(agent, "input_messages", []) or []:
        for part in getattr(msg, "parts", []) or []:
            if isinstance(part, Text) and part.content:
                chunks.extend(flatten_to_strings(part.content))
    attrs = agent.attributes or {}
    prompt_capture = (
        attrs.get("prompt_capture") if isinstance(attrs, Mapping) else None
    )
    if isinstance(prompt_capture, Mapping):
        for key in ("input_context", "input"):
            chunks.extend(flatten_to_strings(prompt_capture.get(key)))
    if isinstance(attrs, Mapping):
        for key in (
            "input_context",
            "input",
            "initial_input",
            "prompt",
            "question",
            "query",
        ):
            chunks.extend(flatten_to_strings(attrs.get(key)))
    return _dedupe_preserve([c.strip() for c in chunks if c])


def _collect_agent_outputs(agent: AgentInvocation) -> list[str]:
    chunks: list[str] = []
    # Extract text from structured output_messages
    for msg in getattr(agent, "output_messages", []) or []:
        for part in getattr(msg, "parts", []) or []:
            if isinstance(part, Text) and part.content:
                chunks.extend(flatten_to_strings(part.content))
    attrs = agent.attributes or {}
    prompt_capture = (
        attrs.get("prompt_capture") if isinstance(attrs, Mapping) else None
    )
    if isinstance(prompt_capture, Mapping):
        for key in ("output_result", "final_output"):
            chunks.extend(flatten_to_strings(prompt_capture.get(key)))
    if isinstance(attrs, Mapping):
        for key in (
            "output_result",
            "final_output",
            "output",
            "response",
            "answer",
            "completion",
            "message",
            "result",
        ):
            chunks.extend(flatten_to_strings(attrs.get(key)))
    return _dedupe_preserve([c.strip() for c in chunks if c])


def is_tool_only_llm(invocation: LLMInvocation) -> bool:
    messages = getattr(invocation, "output_messages", None) or []
    has_text = False
    has_tool = False
    finish_reasons: set[str] = set()
    for msg in messages:
        reason = getattr(msg, "finish_reason", None)
        if isinstance(reason, str):
            finish_reasons.add(reason.lower())
        parts = getattr(msg, "parts", ())
        for part in parts or ():
            if (
                isinstance(part, Text)
                and part.content
                and part.content.strip()
            ):
                has_text = True
                break
            if isinstance(part, ToolCall):
                has_tool = True
            else:
                ptype = getattr(part, "type", None)
                if isinstance(ptype, str) and ptype.lower() == "tool_call":
                    has_tool = True
        if has_text:
            break
    if has_text:
        return False
    implied = getattr(invocation, "response_finish_reasons", None) or ()
    for r in implied:
        if isinstance(r, str):
            finish_reasons.add(r.lower())
    attrs = getattr(invocation, "attributes", None)
    if isinstance(attrs, Mapping):
        attr_reasons = attrs.get("gen_ai.response.finish_reasons")
        if isinstance(attr_reasons, Sequence) and not isinstance(
            attr_reasons, (str, bytes, bytearray)
        ):
            for r in attr_reasons:
                if isinstance(r, str):
                    finish_reasons.add(r.lower())
        elif isinstance(attr_reasons, str):
            finish_reasons.add(attr_reasons.lower())
    if has_tool:
        return True
    if "tool_calls" in finish_reasons:
        return True
    op = getattr(invocation, "operation", None)
    if isinstance(op, str) and op.lower().startswith("execute_tool"):
        return True
    return False


def normalize_invocation(invocation: GenAI) -> CanonicalEvalCase:
    type_name = type(invocation).__name__
    attrs = getattr(invocation, "attributes", None)
    metadata = {
        k: v
        for k, v in (attrs.items() if isinstance(attrs, Mapping) else [])
        if v is not None
    }

    if isinstance(invocation, LLMInvocation):
        input_text = extract_text_from_messages(invocation.input_messages)
        output_text = extract_text_from_messages(invocation.output_messages)
        return CanonicalEvalCase(
            type_name=type_name,
            input_text=input_text,
            output_text=output_text,
            context=extract_context(attrs),
            retrieval_context=extract_retrieval_context(attrs),
            metadata=metadata,
            is_tool_only_llm=is_tool_only_llm(invocation),
            is_agent_non_invoke=False,
        )

    if isinstance(invocation, AgentCreation):
        input_text = "\n\n".join(_collect_agent_inputs(invocation)).strip()
        metadata = {
            "agent_name": getattr(invocation, "name", None),
            "agent_type": getattr(invocation, "agent_type", None),
            **metadata,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return CanonicalEvalCase(
            type_name=type_name,
            input_text=input_text,
            output_text="",
            context=None,
            retrieval_context=extract_retrieval_context(attrs),
            metadata=metadata,
            is_tool_only_llm=False,
            is_agent_non_invoke=True,
        )

    if isinstance(invocation, AgentInvocation):
        input_text = "\n\n".join(_collect_agent_inputs(invocation)).strip()
        output_text = "\n\n".join(_collect_agent_outputs(invocation)).strip()
        context: list[str] | None = None
        if invocation.tools:
            context = ["Tools: " + ", ".join(invocation.tools)]
        metadata = {
            "agent_name": getattr(invocation, "name", None),
            "agent_type": getattr(invocation, "agent_type", None),
            **metadata,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return CanonicalEvalCase(
            type_name=type_name,
            input_text=input_text,
            output_text=output_text,
            context=context,
            retrieval_context=extract_retrieval_context(attrs),
            metadata=metadata,
            is_tool_only_llm=False,
            is_agent_non_invoke=getattr(invocation, "operation", None)
            != "invoke_agent",
        )

    # Fallback for other types
    return CanonicalEvalCase(
        type_name=type_name,
        input_text="",
        output_text="",
        context=None,
        retrieval_context=None,
        metadata=metadata,
        is_tool_only_llm=False,
        is_agent_non_invoke=False,
    )


__all__ = [
    "CanonicalEvalCase",
    "normalize_invocation",
    "extract_text_from_messages",
    "extract_context",
    "extract_retrieval_context",
    "flatten_to_strings",
    "is_tool_only_llm",
]
