"""Simplified LangChain callback handler (Phase 1).

Only maps callbacks to GenAI util types and delegates lifecycle to TelemetryHandler.
Complex logic removed (agent heuristics, child counting, prompt capture, events).
"""

from __future__ import annotations

import json
from typing import Any, Optional, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Workflow,
    Step,
    AgentInvocation,
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    Error as GenAIError,
)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


def _serialize(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return str(obj)
        except (TypeError, ValueError):
            return None


def _resolve_agent_name(
    tags: Optional[list[str]], metadata: Optional[dict[str, Any]]
) -> Optional[str]:
    if metadata:
        for key in ("agent_name", "gen_ai.agent.name", "agent"):
            value = metadata.get(key)
            if value:
                return _safe_str(value)
    if tags:
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag_value = tag.strip()
            lower_value = tag_value.lower()
            if lower_value.startswith("agent:") and len(tag_value) > 6:
                return _safe_str(tag_value.split(":", 1)[1])
            if lower_value.startswith("agent_") and len(tag_value) > 6:
                return _safe_str(tag_value.split("_", 1)[1])
            if lower_value == "agent":
                return _safe_str(tag_value)
    return None


def _is_agent_root(
    tags: Optional[list[str]], metadata: Optional[dict[str, Any]]
) -> bool:
    if _resolve_agent_name(tags, metadata):
        return True
    if tags:
        for tag in tags:
            try:
                if "agent" in str(tag).lower():
                    return True
            except (TypeError, ValueError):
                continue
    if metadata and metadata.get("agent_name"):
        return True
    return False


def _extract_tool_details(
    metadata: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not metadata:
        return None

    tool_data: dict[str, Any] = {}
    nested = metadata.get("gen_ai.tool")
    if isinstance(nested, dict):
        tool_data.update(nested)

    detection_flag = bool(tool_data)
    for key, value in list(metadata.items()):
        if not isinstance(key, str):
            continue
        lower_key = key.lower()
        if lower_key.startswith("gen_ai.tool."):
            suffix = key.split(".", 2)[-1]
            tool_data[suffix] = value
            detection_flag = True
            continue
        if lower_key in {
            "tool_name",
            "tool_id",
            "tool_call_id",
            "tool_args",
            "tool_arguments",
            "tool_input",
            "tool_parameters",
        }:
            name_parts = lower_key.split("_", 1)
            suffix = name_parts[-1] if len(name_parts) > 1 else lower_key
            tool_data[suffix] = value
            detection_flag = True

    for hint_key in (
        "gen_ai.step.type",
        "step_type",
        "type",
        "run_type",
        "langchain_run_type",
    ):
        hint_val = metadata.get(hint_key)
        if isinstance(hint_val, str) and "tool" in hint_val.lower():
            detection_flag = True
            break

    if not detection_flag:
        return None

    name_value = tool_data.get("name") or metadata.get("gen_ai.step.name")
    if not name_value:
        return None

    arguments = tool_data.get("arguments")
    if arguments is None:
        for candidate in ("input", "args", "parameters"):
            if candidate in tool_data:
                arguments = tool_data[candidate]
                break

    tool_id = tool_data.get("id") or tool_data.get("call_id")
    if tool_id is not None:
        tool_id = _safe_str(tool_id)

    return {
        "name": _safe_str(name_value),
        "arguments": arguments,
        "id": tool_id,
    }


class LangchainCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        telemetry_handler: Optional[TelemetryHandler] = None,
    ) -> None:
        super().__init__()
        self._handler = telemetry_handler

    def _find_nearest_agent(self, run_id: Optional[UUID]) -> Optional[AgentInvocation]:
        current = run_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._handler.get_entity(current)
            if isinstance(entity, AgentInvocation):
                return entity
            if entity is None:
                break
            current = getattr(entity, "parent_run_id", None)
        return None

    def _start_agent_invocation(
        self,
        *,
        name: str,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        attrs: dict[str, Any],
        inputs: dict[str, Any],
        metadata: Optional[dict[str, Any]],
        agent_name: Optional[str],
    ) -> AgentInvocation:
        agent = AgentInvocation(
            name=name,
            run_id=run_id,
            attributes=attrs,
        )
        agent.input_context = _serialize(inputs)
        agent.agent_name = _safe_str(agent_name) if agent_name else name
        agent.parent_run_id = parent_run_id
        agent.framework = "langchain"
        if metadata:
            if metadata.get("agent_type"):
                agent.agent_type = _safe_str(metadata["agent_type"])
            if metadata.get("model_name"):
                agent.model = _safe_str(metadata["model_name"])
            if metadata.get("system"):
                agent.system = _safe_str(metadata["system"])
        self._handler.start_agent(agent)
        return agent

    def on_chain_start(
        self,
        serialized: Optional[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "chain")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        agent_name_hint = _resolve_agent_name(tags, metadata)
        if parent_run_id is None:
            if _is_agent_root(tags, metadata):
                self._start_agent_invocation(
                    name=name,
                    run_id=run_id,
                    parent_run_id=None,
                    attrs=attrs,
                    inputs=inputs,
                    metadata=metadata,
                    agent_name=agent_name_hint,
                )
            else:
                wf = Workflow(name=name, run_id=run_id, attributes=attrs)
                wf.initial_input = _serialize(inputs)
                self._handler.start_workflow(wf)
            return
        else:
            context_agent = self._find_nearest_agent(parent_run_id)
            context_agent_name = (
                _safe_str(context_agent.agent_name or context_agent.name)
                if context_agent
                else None
            )
            if agent_name_hint:
                hint_normalized = agent_name_hint.lower()
                context_normalized = (
                    context_agent_name.lower() if context_agent_name else None
                )
                if context_normalized != hint_normalized:
                    self._start_agent_invocation(
                        name=name,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        attrs=attrs,
                        inputs=inputs,
                        metadata=metadata,
                        agent_name=agent_name_hint,
                    )
                    return
            tool_info = _extract_tool_details(metadata)
            if tool_info is not None:
                existing = self._handler.get_entity(run_id)
                if isinstance(existing, ToolCall):
                    tool = existing
                    if context_agent is not None:
                        agent_name_value = (
                            context_agent.agent_name or context_agent.name
                        )
                        if not getattr(tool, "agent_name", None):
                            tool.agent_name = _safe_str(agent_name_value)
                        if not getattr(tool, "agent_id", None):
                            tool.agent_id = str(context_agent.run_id)
                else:
                    arguments = tool_info.get("arguments")
                    if arguments is None:
                        arguments = inputs
                    tool = ToolCall(
                        name=tool_info.get("name", name),
                        id=tool_info.get("id"),
                        arguments=arguments,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        attributes=attrs,
                    )
                    tool.framework = "langchain"
                    if context_agent is not None and context_agent_name is not None:
                        tool.agent_name = context_agent_name
                        tool.agent_id = str(context_agent.run_id)
                    self._handler.start_tool_call(tool)
                if inputs is not None and getattr(tool, "arguments", None) is None:
                    tool.arguments = inputs
                if getattr(tool, "arguments", None) is not None:
                    serialized_args = _serialize(tool.arguments)
                    if serialized_args is not None:
                        tool.attributes.setdefault("tool.arguments", serialized_args)
            else:
                step = Step(
                    name=name,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    step_type="chain",
                    attributes=attrs,
                )
                if context_agent is not None:
                    if context_agent_name is not None:
                        step.agent_name = context_agent_name
                    step.agent_id = str(context_agent.run_id)
                step.input_data = _serialize(inputs)
                self._handler.start_step(step)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        entity = self._handler.get_entity(run_id)
        if entity is None:
            return
        if isinstance(entity, Workflow):
            entity.final_output = _serialize(outputs)
            self._handler.stop_workflow(entity)
        elif isinstance(entity, AgentInvocation):
            entity.output_result = _serialize(outputs)
            self._handler.stop_agent(entity)
        elif isinstance(entity, Step):
            entity.output_data = _serialize(outputs)
            self._handler.stop_step(entity)
        elif isinstance(entity, ToolCall):
            serialized = _serialize(outputs)
            if serialized is not None:
                entity.attributes.setdefault("tool.response", serialized)
            self._handler.stop_tool_call(entity)

    def on_chat_model_start(
        self,
        serialized: Optional[dict[str, Any]],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        model_source = (
            payload.get("name")
            or payload.get("id")
            or (metadata.get("model_name") if metadata else None)
            or extra.get("name")
        )
        request_model = _safe_str(model_source or "model")
        input_messages: list[InputMessage] = []
        for batch in messages:
            for m in batch:
                content = getattr(m, "content", "")
                input_messages.append(
                    InputMessage(role="user", parts=[Text(content=_safe_str(content))])
                )
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        inv = LLMInvocation(
            request_model=request_model,
            input_messages=input_messages,
            attributes=attrs,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        if parent_run_id is not None:
            context_agent = self._find_nearest_agent(parent_run_id)
            if context_agent is not None:
                agent_name_value = context_agent.agent_name or context_agent.name
                inv.agent_name = _safe_str(agent_name_value)
                inv.agent_id = str(context_agent.run_id)
        self._handler.start_llm(inv)

    def on_llm_start(
        self,
        serialized: Optional[dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        message_batches = [[HumanMessage(content=p)] for p in prompts]
        self.on_chat_model_start(
            serialized=serialized,
            messages=message_batches,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **extra,
        )
        inv = self._handler.get_entity(run_id)
        if isinstance(inv, LLMInvocation):
            inv.operation = "generate_text"

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        inv = self._handler.get_entity(run_id)
        if not isinstance(inv, LLMInvocation):
            return
        generations = getattr(response, "generations", [])
        content = None
        if generations and generations[0] and generations[0][0].message:
            content = getattr(generations[0][0].message, "content", None)
        if content is not None:
            finish_reason = (
                generations[0][0].generation_info.get("finish_reason")
                if generations[0][0].generation_info
                else None
            )
            if finish_reason == "tool_calls":
                inv.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=["ToolCall"],
                        finish_reason=finish_reason or "tool_calls",
                    )
                ]
            else:
                inv.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=_safe_str(content))],
                        finish_reason=finish_reason or "stop",
                    )
                ]
        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        inv.input_tokens = usage.get("prompt_tokens")
        inv.output_tokens = usage.get("completion_tokens")
        self._handler.stop_llm(inv)

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "tool")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        context_agent = (
            self._find_nearest_agent(parent_run_id)
            if parent_run_id is not None
            else None
        )
        context_agent_name = (
            _safe_str(context_agent.agent_name or context_agent.name)
            if context_agent
            else None
        )
        id_source = payload.get("id") or extra.get("id")
        if isinstance(id_source, (list, tuple)):
            id_value = ".".join(_safe_str(part) for part in id_source)
        elif id_source is not None:
            id_value = _safe_str(id_source)
        else:
            id_value = None
        arguments: Any = inputs if inputs is not None else input_str
        existing = self._handler.get_entity(run_id)
        if isinstance(existing, ToolCall):
            if arguments is not None:
                existing.arguments = arguments
            if attrs:
                existing.attributes.update(attrs)
            if context_agent is not None:
                if (
                    not getattr(existing, "agent_name", None)
                    and context_agent_name is not None
                ):
                    existing.agent_name = context_agent_name
                if not getattr(existing, "agent_id", None):
                    existing.agent_id = str(context_agent.run_id)
            if existing.framework is None:
                existing.framework = "langchain"
            return
        tool = ToolCall(
            name=name,
            id=id_value,
            arguments=arguments,
            run_id=run_id,
            parent_run_id=parent_run_id,
            attributes=attrs,
        )
        tool.framework = "langchain"
        if context_agent is not None and context_agent_name is not None:
            tool.agent_name = context_agent_name
            tool.agent_id = str(context_agent.run_id)
        if arguments is not None:
            serialized_args = _serialize(arguments)
            if serialized_args is not None:
                tool.attributes.setdefault("tool.arguments", serialized_args)
        if inputs is None and input_str:
            tool.attributes.setdefault("tool.input_str", _safe_str(input_str))
        self._handler.start_tool_call(tool)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        tool = self._handler.get_entity(run_id)
        if not isinstance(tool, ToolCall):
            return
        serialized = _serialize(output)
        if serialized is not None:
            tool.attributes.setdefault("tool.response", serialized)
        self._handler.stop_tool_call(tool)

    def _fail(self, run_id: UUID, error: BaseException) -> None:
        self._handler.fail_by_run_id(
            run_id, GenAIError(message=str(error), type=type(error))
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> None:
        self._fail(run_id, error)
