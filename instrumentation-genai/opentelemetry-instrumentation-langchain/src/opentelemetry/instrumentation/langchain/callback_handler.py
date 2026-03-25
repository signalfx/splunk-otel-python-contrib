"""Simplified LangChain callback handler (Phase 1).

Only maps callbacks to GenAI util types and delegates lifecycle to TelemetryHandler.
Complex logic removed (agent heuristics, child counting, prompt capture, events).
"""

from __future__ import annotations

import json
from typing import Any, Optional, List, Dict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.outputs import LLMResult
from opentelemetry.util.genai.handler import (
    GenAIContext,
    TelemetryHandler,
    get_genai_context,
    set_genai_context,
)
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
    ErrorClassification,
)
from opentelemetry.util.genai.attributes import (
    GEN_AI_COMMAND,
    GEN_AI_FINISH_REASON,
    FINISH_REASON_INTERRUPTED,
)

# Error type names that indicate flow-control, not real errors.
# Uses type name strings to avoid importing LangGraph at instrumentation time.
_INTERRUPT_TYPE_NAMES = frozenset(
    {
        "GraphInterrupt",
        "NodeInterrupt",
        "Interrupt",
    }
)
_CANCELLATION_TYPE_NAMES = frozenset(
    {
        "CancelledError",
        "TaskCancelledError",
    }
)


def _classify_error(error: BaseException) -> ErrorClassification:
    """Classify an exception as a real error, interrupt, or cancellation."""
    for cls in type(error).__mro__:
        if cls.__name__ in _INTERRUPT_TYPE_NAMES:
            return ErrorClassification.INTERRUPT
        if cls.__name__ in _CANCELLATION_TYPE_NAMES:
            return ErrorClassification.CANCELLATION
    return ErrorClassification.REAL_ERROR


def _extract_interrupt_values(error: BaseException) -> list[Any] | None:
    """Extract the interrupt value(s) from a GraphInterrupt exception.

    ``GraphInterrupt.__init__`` receives a sequence of ``Interrupt`` objects
    stored as ``args[0]``.  Each ``Interrupt`` carries a ``.value`` attribute
    that holds the payload the node passed to ``interrupt()``.

    Returns a list of extracted values, or ``None`` when nothing useful can
    be extracted.
    """
    if not (hasattr(error, "args") and error.args):
        return None
    interrupts = error.args[0]
    if not isinstance(interrupts, (list, tuple)):
        return None
    values: list[Any] = []
    for item in interrupts:
        val = getattr(item, "value", None)
        if val is not None:
            values.append(val)
    return values or None


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


def _serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string.

    Uses default=str to handle non-JSON-serializable objects (like LangChain
    message objects) by converting them to their string representation while
    keeping the overall structure as valid JSON.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _make_command_input_message(command: Any) -> list[InputMessage]:
    """Create input messages from a LangGraph Command object.

    The ``resume`` value is the human's response to an interrupt question,
    so it maps naturally to a user message.  For non-resume commands we
    fall back to the string representation of the Command.
    """
    resume = getattr(command, "resume", None)
    if resume is not None:
        content = (
            json.dumps(resume, ensure_ascii=False, default=str)
            if not isinstance(resume, str)
            else resume
        )
        return [InputMessage(role="user", parts=[Text(content)])]
    return [InputMessage(role="user", parts=[Text(_safe_str(command))])]


def _make_input_message(data: dict[str, Any]) -> list[InputMessage]:
    """Create structured input message with full data as JSON."""
    input_messages: list[InputMessage] = []
    messages = data.get("messages")
    if messages is None:
        return []
    for msg in messages:
        content = getattr(msg, "content", "")
        if content:
            # TODO: for invoke_agent type invocation, when system_messages is added, can filter SystemMessage separately if needed and only add here HumanMessage, currently all messages are added
            input_message = InputMessage(role="user", parts=[Text(_safe_str(content))])
            input_messages.append(input_message)
    return input_messages


def _make_output_message(data: dict[str, Any]) -> list[OutputMessage]:
    """Create structured output message with full data as JSON."""
    output_messages: list[OutputMessage] = []
    messages = data.get("messages")
    if messages is None:
        return []
    for msg in messages:
        content = getattr(msg, "content", "")
        if content:
            if isinstance(msg, AIMessage):
                output_message = OutputMessage(
                    role="assistant", parts=[Text(_safe_str(msg.content))]
                )
                output_messages.append(output_message)
    return output_messages


def _make_last_output_message(data: dict[str, Any]) -> list[OutputMessage]:
    """Extract only the last AI message as the output.

    For Workflow and AgentInvocation spans, the final AI message best represents
    the actual output. Intermediate AI messages (e.g., tool-call decisions) are
    already captured in child LLM invocation spans.
    """
    all_messages = _make_output_message(data)
    if all_messages:
        return [all_messages[-1]]
    return []


def _make_workflow_output_fallback(data: dict[str, Any]) -> Optional[str]:
    """Create output summary from non-message state fields.

    Fallback for when workflow output doesn't contain AI messages.
    This is common in LangGraph where agent nodes update structured
    state fields rather than the message list.
    """
    if not data:
        return None
    # Exclude messages and internal fields that don't represent output
    exclude_keys = {"messages", "intermediate_steps"}
    output_data = {
        k: v for k, v in data.items() if k not in exclude_keys and v is not None
    }
    if not output_data:
        return None
    return _serialize(output_data)


def _infer_conversation_id(metadata: dict[str, Any] | None) -> str | None:
    """Extract conversation_id from LangChain/LangGraph metadata.

    Checks well-known keys in priority order:
    1. ``conversation_id`` — explicit GenAI convention
    2. ``thread_id`` — LangGraph's native session key

    Returns:
        The conversation identifier as a string, or ``None``.
    """
    if not metadata:
        return None
    for key in ("conversation_id", "thread_id"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
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
            # Don't return "agent" itself as a name - it's just a marker tag
    return None


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


def _agent_span_id(agent: AgentInvocation) -> Optional[str]:
    """Return the agent's span ID as a hex string, if available."""
    if agent.span_id is not None:
        return f"{agent.span_id:016x}"
    return None


class _InvocationManager:
    """Local store mapping LangChain run_ids to GenAI invocation objects.

    Keyed by LangChain's own run_id UUID (from callback API), not the
    GenAI run_id field. Tracks parent-child relationships so the parent
    chain can be walked without a central registry.
    """

    def __init__(self) -> None:
        self._invocations: Dict[UUID, Any] = {}
        self._parents: Dict[UUID, Optional[UUID]] = {}
        self._root_run_id: Optional[UUID] = None  # Track the root entity

    def add(self, run_id: UUID, parent_run_id: Optional[UUID], invocation: Any) -> None:
        self._invocations[run_id] = invocation
        self._parents[run_id] = parent_run_id
        # Track the root (first entity with no parent)
        if parent_run_id is None and self._root_run_id is None:
            self._root_run_id = run_id

    def get(self, run_id: UUID) -> Any:
        return self._invocations.get(run_id)

    def get_parent_id(self, run_id: UUID) -> Optional[UUID]:
        return self._parents.get(run_id)

    def get_root(self) -> Any:
        """Get the root entity (Workflow or AgentInvocation)."""
        if self._root_run_id is not None:
            return self._invocations.get(self._root_run_id)
        return None

    def remove(self, run_id: UUID) -> None:
        self._invocations.pop(run_id, None)
        self._parents.pop(run_id, None)
        # Clear root tracking when root is removed
        if run_id == self._root_run_id:
            self._root_run_id = None


class LangchainCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        telemetry_handler: Optional[TelemetryHandler] = None,
    ) -> None:
        super().__init__()
        self._handler = telemetry_handler
        self._invocation_manager = _InvocationManager()
        # Tracks ContextVar state before we push an inferred conversation_id
        # so it can be restored when the root entity finishes.
        self._inferred_context_prev: dict[UUID, GenAIContext | None] = {}

    def _restore_inferred_context(self, run_id: UUID) -> None:
        """Restore ContextVar state pushed in on_chain_start for inferred conversation_id."""
        prev = self._inferred_context_prev.pop(run_id, None)
        if prev is not None:
            set_genai_context(
                conversation_id=prev.conversation_id,
                properties=prev.properties if prev.properties else None,
            )

    def _resolve_parent_span(self, parent_run_id: Optional[UUID]):
        """Look up the parent invocation's span for explicit parent-child linking."""
        if parent_run_id is None:
            return None
        parent_entity = self._invocation_manager.get(parent_run_id)
        if parent_entity is not None:
            return getattr(parent_entity, "span", None)
        return None

    def _find_nearest_agent(self, run_id: Optional[UUID]) -> Optional[AgentInvocation]:
        current = run_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._invocation_manager.get(current)
            if isinstance(entity, AgentInvocation):
                return entity
            if entity is None:
                break
            current = self._invocation_manager.get_parent_id(current)
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
        conversation_id: Optional[str] = None,
        command_input_messages: Optional[list[InputMessage]] = None,
    ) -> AgentInvocation:
        agent = AgentInvocation(
            name=name,
            attributes=attrs,
        )
        agent.input_messages = command_input_messages or _make_input_message(inputs)
        agent.agent_name = _safe_str(agent_name) if agent_name else name
        agent.framework = "langchain"
        if conversation_id:
            agent.conversation_id = conversation_id
        if metadata:
            if metadata.get("agent_type"):
                agent.agent_type = _safe_str(metadata["agent_type"])
            if metadata.get("model_name"):
                agent.model = _safe_str(metadata["model_name"])
            if metadata.get("system"):
                agent.system = _safe_str(metadata["system"])
        agent.parent_span = self._resolve_parent_span(parent_run_id)
        self._handler.start_agent(agent)
        self._invocation_manager.add(run_id, parent_run_id, agent)
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
        name_source = (
            payload.get("name")
            or payload.get("id")
            or extra.get("name")
            or (metadata.get("langgraph_node") if metadata else None)
        )
        name = _safe_str(name_source or "chain")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        agent_name_hint = _resolve_agent_name(tags, metadata)

        # LangGraph passes a Command object (not a dict) as inputs on
        # resume.  Detect this for resume signalling, capture a
        # meaningful input representation, then normalise to a dict so
        # downstream helpers (_make_input_message, etc.) work.
        is_command = type(inputs).__name__ == "Command"
        command_input_messages: list[InputMessage] = []
        if is_command:
            command_input_messages = _make_command_input_message(inputs)
            inputs = {}

        if parent_run_id is None:
            # Infer conversation_id from metadata (thread_id) only when
            # no explicit genai_context(conversation_id=...) is active.
            # Priority: explicit genai_context > inferred metadata > none.
            inferred_conv_id: str | None = None
            ctx = get_genai_context()
            if not ctx.conversation_id:
                inferred_conv_id = _infer_conversation_id(metadata)

            # Push the inferred conversation_id into the ContextVar so
            # _apply_genai_context() propagates it to ALL child entities
            # (LLM calls, tools, steps, nested agents).  Save previous
            # state so we can restore it when the root entity finishes.
            if inferred_conv_id:
                self._inferred_context_prev[run_id] = ctx
                set_genai_context(conversation_id=inferred_conv_id)

            # Detect resume: Command input or checkpoint_id in metadata.
            is_resume = is_command or bool(
                metadata and metadata.get("checkpoint_id") is not None
            )

            # Determine root span type: AgentInvocation (default) or Workflow.
            # Workflow is used only when explicitly requested via env var or
            # workflow_name in LangGraph config metadata.
            #
            # Note: the old _is_agent_root(tags, metadata) check has been
            # removed from the root path.  Root spans are now always
            # AgentInvocation by default, regardless of whether agent
            # tags/metadata are present.  _is_agent_root still governs
            # agent-promotion for *child* chains.
            workflow_name_override = metadata.get("workflow_name") if metadata else None
            force_workflow = self._handler.should_use_workflow_root(
                workflow_name=workflow_name_override
            )

            if force_workflow:
                wf = Workflow(name=workflow_name_override or name, attributes=attrs)
                wf.input_messages = command_input_messages or _make_input_message(
                    inputs
                )
                if inferred_conv_id:
                    wf.conversation_id = inferred_conv_id
                if is_resume:
                    wf.attributes[GEN_AI_COMMAND] = "resume"
                self._handler.start_workflow(wf)
                self._invocation_manager.add(run_id, None, wf)
            else:
                agent_name = agent_name_hint or name
                if is_resume:
                    attrs[GEN_AI_COMMAND] = "resume"
                self._start_agent_invocation(
                    name=name,
                    run_id=run_id,
                    parent_run_id=None,
                    attrs=attrs,
                    inputs=inputs,
                    metadata=metadata,
                    agent_name=agent_name,
                    conversation_id=inferred_conv_id,
                    command_input_messages=command_input_messages or None,
                )
            return
        else:
            # Skip if parent entity no longer exists (e.g., LangGraph
            # replays the interrupted node during resume — the parent
            # workflow from the previous trace was already ended).
            # TODO: _invocation_manager is per-handler instance; consider
            #  making LangchainCallbackHandler a singleton so all
            #  invocations share the same manager.
            if self._invocation_manager.get(parent_run_id) is None:
                return

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
                existing = self._invocation_manager.get(run_id)
                if isinstance(existing, ToolCall):
                    tool = existing
                    if context_agent is not None:
                        agent_name_value = (
                            context_agent.agent_name or context_agent.name
                        )
                        if not getattr(tool, "agent_name", None):
                            tool.agent_name = _safe_str(agent_name_value)
                        if not getattr(tool, "agent_id", None):
                            tool.agent_id = _agent_span_id(context_agent)
                else:
                    # Filter out tool-specific metadata from attributes
                    # since they're stored in dedicated fields
                    tool_attrs = {
                        k: v
                        for k, v in attrs.items()
                        if not (
                            isinstance(k, str) and k.lower().startswith("gen_ai.tool.")
                        )
                    }
                    arguments = tool_info.get("arguments")
                    if arguments is None:
                        arguments = inputs
                    tool = ToolCall(
                        name=tool_info.get("name", name),
                        id=tool_info.get("id"),
                        arguments=arguments,
                        attributes=tool_attrs,
                    )
                    tool.framework = "langchain"
                    if context_agent is not None and context_agent_name is not None:
                        tool.agent_name = context_agent_name
                        tool.agent_id = _agent_span_id(context_agent)
                    tool.parent_span = self._resolve_parent_span(parent_run_id)
                    self._handler.start_tool_call(tool)
                    self._invocation_manager.add(run_id, parent_run_id, tool)
                if inputs is not None and getattr(tool, "arguments", None) is None:
                    tool.arguments = inputs
                if getattr(tool, "arguments", None) is not None:
                    serialized_args = _serialize(tool.arguments)
                    if serialized_args is not None:
                        tool.attributes.setdefault("tool.arguments", serialized_args)
            else:
                step = Step(
                    name=name,
                    step_type="chain",
                    attributes=attrs,
                )
                if context_agent is not None:
                    if context_agent_name is not None:
                        step.agent_name = context_agent_name
                    step.agent_id = _agent_span_id(context_agent)
                step.parent_span = self._resolve_parent_span(parent_run_id)
                self._handler.start_step(step)
                self._invocation_manager.add(run_id, parent_run_id, step)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        entity = self._invocation_manager.get(run_id)
        if entity is None:
            return

        self._restore_inferred_context(run_id)
        if isinstance(entity, (Workflow, AgentInvocation)):
            # When the entity was interrupted, _fail() already set the
            # interrupt value as the output message.  Don't overwrite it
            # with the full graph state dump that LangGraph passes here.
            already_interrupted = (
                entity.attributes.get(GEN_AI_FINISH_REASON) == FINISH_REASON_INTERRUPTED
            )
            if not already_interrupted:
                output_msgs = _make_last_output_message(outputs)
                if output_msgs:
                    entity.output_messages = output_msgs
                else:
                    # Fallback: serialize non-message state fields as output.
                    # Common in LangGraph where nodes update structured state
                    # fields rather than the message list.
                    fallback = _make_workflow_output_fallback(outputs)
                    if fallback:
                        entity.output_messages = [
                            OutputMessage(role="assistant", parts=[Text(fallback)])
                        ]
            self._handler.finish(entity)
        elif isinstance(entity, Step):
            self._handler.stop_step(entity)
        elif isinstance(entity, ToolCall):
            serialized = _serialize(outputs)
            if serialized is not None:
                entity.attributes.setdefault("tool.response", serialized)
            self._handler.stop_tool_call(entity)
        self._invocation_manager.remove(run_id)

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
        invocation_params = extra.get("invocation_params", {})
        # Priority: invocation_params.model_name > metadata.model_name.
        # Removed fallback to payload name/id to avoid misreporting implementation class as model.
        model_source = invocation_params.get("model_name") or (
            metadata.get("model_name") if metadata else None
        )
        request_model = _safe_str(model_source or "unknown_model")
        input_messages: list[InputMessage] = []
        for batch in messages:
            for m in batch:
                content = getattr(m, "content", "")
                input_messages.append(
                    InputMessage(role="user", parts=[Text(content=_safe_str(content))])
                )

        # Build attributes from metadata and invocation_params
        attrs: dict[str, Any] = {}

        # Process metadata - skip ls_* fields as they're extracted to dedicated fields
        if metadata:
            for key, value in metadata.items():
                if isinstance(key, str) and not key.startswith("ls_"):
                    attrs[key] = value

        # Add tags
        if tags:
            attrs["tags"] = [str(t) for t in tags]

        # Process invocation_params - add with request_ prefix
        if invocation_params:
            # Standard params get request_ prefix
            for key in (
                "top_p",
                "seed",
                "temperature",
                "frequency_penalty",
                "presence_penalty",
            ):
                if key in invocation_params:
                    attrs[f"request_{key}"] = invocation_params[key]

            # Handle nested model_kwargs
            if "model_kwargs" in invocation_params:
                attrs["model_kwargs"] = invocation_params["model_kwargs"]
                # Also check for max_tokens in model_kwargs
                mk = invocation_params["model_kwargs"]
                if isinstance(mk, dict) and "max_tokens" in mk:
                    attrs["request_max_tokens"] = mk["max_tokens"]

        # Handle max_tokens from metadata.ls_max_tokens
        if (
            metadata
            and "ls_max_tokens" in metadata
            and "request_max_tokens" not in attrs
        ):
            attrs["request_max_tokens"] = metadata["ls_max_tokens"]

        # Add callback info from serialized
        if payload.get("name"):
            attrs["callback.name"] = payload["name"]
        if payload.get("id"):
            attrs["callback.id"] = payload["id"]

        # Set provider from ls_provider in metadata
        provider = None
        if metadata and "ls_provider" in metadata:
            provider = _safe_str(metadata["ls_provider"])

        inv = LLMInvocation(
            request_model=request_model,
            input_messages=input_messages,
            attributes=attrs,
        )
        if provider:
            inv.provider = provider
        if parent_run_id is not None:
            context_agent = self._find_nearest_agent(parent_run_id)
            if context_agent is not None:
                agent_name_value = context_agent.agent_name or context_agent.name
                inv.agent_name = _safe_str(agent_name_value)
                inv.agent_id = _agent_span_id(context_agent)
        inv.parent_span = self._resolve_parent_span(parent_run_id)
        self._handler.start_llm(inv)
        self._invocation_manager.add(run_id, parent_run_id, inv)

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
        inv = self._invocation_manager.get(run_id)
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
        inv = self._invocation_manager.get(run_id)
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

        # Extract response model from response metadata if available
        if not inv.response_model_name and generations:
            for generation_list in generations:
                for generation in generation_list:
                    if (
                        hasattr(generation, "message")
                        and hasattr(generation.message, "response_metadata")
                        and generation.message.response_metadata
                    ):
                        model_name = generation.message.response_metadata.get(
                            "model_name"
                        )
                        if model_name:
                            inv.response_model_name = _safe_str(model_name)
                            break
                if inv.response_model_name:
                    break

        self._handler.stop_llm(inv)
        self._invocation_manager.remove(run_id)

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
        existing = self._invocation_manager.get(run_id)
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
                    existing.agent_id = _agent_span_id(context_agent)
            if existing.framework is None:
                existing.framework = "langchain"
            return
        tool = ToolCall(
            name=name,
            id=id_value,
            arguments=arguments,
            attributes=attrs,
        )
        tool.framework = "langchain"
        if context_agent is not None and context_agent_name is not None:
            tool.agent_name = context_agent_name
            tool.agent_id = _agent_span_id(context_agent)
        tool.parent_span = self._resolve_parent_span(parent_run_id)
        if arguments is not None:
            serialized_args = _serialize(arguments)
            if serialized_args is not None:
                tool.attributes.setdefault("tool.arguments", serialized_args)
        if inputs is None and input_str:
            tool.attributes.setdefault("tool.input_str", _safe_str(input_str))
        self._handler.start_tool_call(tool)
        self._invocation_manager.add(run_id, parent_run_id, tool)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        tool = self._invocation_manager.get(run_id)
        if not isinstance(tool, ToolCall):
            return
        serialized = _serialize(output)
        if serialized is not None:
            tool.attributes.setdefault("tool.response", serialized)
        self._handler.stop_tool_call(tool)
        self._invocation_manager.remove(run_id)

    def _fail(self, run_id: UUID, error: BaseException) -> None:
        classification = _classify_error(error)
        entity = self._invocation_manager.get(run_id)
        if entity is None:
            return
        genai_error = GenAIError(
            message=str(error),
            type=type(error),
            classification=classification,
        )

        # Bubble up "interrupted" status to the root span when an interrupt occurs
        # on any child entity. This ensures the root span reflects that the
        # conversation was interrupted, even if the interrupt happened deeper
        # in the call hierarchy.
        if classification == ErrorClassification.INTERRUPT:
            # Extract the clean interrupt value(s) from the
            # GraphInterrupt exception.  The value is what the node
            # passed to ``interrupt()`` — e.g. the question or review
            # payload.
            interrupt_values = _extract_interrupt_values(error)
            if interrupt_values is not None:
                raw_value = (
                    interrupt_values[0]
                    if len(interrupt_values) == 1
                    else interrupt_values
                )
                content = (
                    json.dumps(raw_value, ensure_ascii=False, default=str)
                    if not isinstance(raw_value, str)
                    else raw_value
                )
                # Use the clean interrupt value as the error message so
                # _apply_error_status writes a human-readable
                # finish_reason_description on the child step span.
                genai_error = GenAIError(
                    message=content,
                    type=type(error),
                    classification=classification,
                )
            root_entity = self._invocation_manager.get_root()
            if root_entity is not None and root_entity is not entity:
                root_entity.attributes[GEN_AI_FINISH_REASON] = FINISH_REASON_INTERRUPTED
                # Set the interrupt value as the root output message.
                # The finish_reason_description is intentionally NOT set
                # on the root — it would duplicate gen_ai.output.messages.
                if interrupt_values is not None:
                    root_entity.output_messages = [
                        OutputMessage(role="assistant", parts=[Text(content)])
                    ]

        self._handler.fail(entity, genai_error)
        self._invocation_manager.remove(run_id)

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
        self._restore_inferred_context(run_id)
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
