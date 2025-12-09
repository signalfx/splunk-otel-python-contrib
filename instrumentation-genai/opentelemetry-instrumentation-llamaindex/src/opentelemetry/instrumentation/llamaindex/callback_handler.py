from typing import Any, Dict, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType

from opentelemetry import trace
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    EmbeddingInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
)

from .vendor_detection import detect_vendor_from_class


def _safe_str(value: Any) -> str:
    """Safely convert value to string."""
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


class LlamaindexCallbackHandler(BaseCallbackHandler):
    """LlamaIndex callback handler supporting LLM and Embedding instrumentation."""

    def __init__(
        self,
        telemetry_handler: Optional[TelemetryHandler] = None,
    ) -> None:
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._handler = telemetry_handler

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace - required by BaseCallbackHandler."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a trace - required by BaseCallbackHandler."""
        pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Handle event start - processing LLM, EMBEDDING, AGENT_STEP, and FUNCTION_CALL events."""
        if event_type == CBEventType.LLM:
            self._handle_llm_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.EMBEDDING:
            self._handle_embedding_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.AGENT_STEP:
            self._handle_agent_step_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.FUNCTION_CALL:
            self._handle_function_call_start(event_id, parent_id, payload, **kwargs)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Handle event end - processing LLM, EMBEDDING, AGENT_STEP, and FUNCTION_CALL events."""
        if event_type == CBEventType.LLM:
            self._handle_llm_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.EMBEDDING:
            self._handle_embedding_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.AGENT_STEP:
            self._handle_agent_step_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.FUNCTION_CALL:
            self._handle_function_call_end(event_id, payload, **kwargs)

    def _handle_llm_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM invocation start."""
        if not self._handler or not payload:
            return

        # Extract model information from payload
        serialized = payload.get("serialized", {})
        model_name = (
            serialized.get("model")
            or serialized.get("model_name")
            or "unknown"
        )

        # Extract messages from payload
        # LlamaIndex messages are ChatMessage objects with .content and .role properties
        messages = payload.get("messages", [])
        input_messages = []

        for msg in messages:
            # Handle ChatMessage objects (has .content property and .role attribute)
            if hasattr(msg, "content") and hasattr(msg, "role"):
                # Extract role - could be MessageRole enum
                role_value = (
                    str(msg.role.value)
                    if hasattr(msg.role, "value")
                    else str(msg.role)
                )
                # Extract content - this is a property that pulls from blocks[0].text
                content = _safe_str(msg.content)
                input_messages.append(
                    InputMessage(
                        role=role_value, parts=[Text(content=content)]
                    )
                )
            elif isinstance(msg, dict):
                # Handle serialized messages (dict format)
                role = msg.get("role", "user")
                # Try to extract from blocks first (LlamaIndex format)
                blocks = msg.get("blocks", [])
                if blocks and isinstance(blocks[0], dict):
                    content = blocks[0].get("text", "")
                else:
                    # Fallback to direct content field
                    content = msg.get("content", "")

                role_value = (
                    str(role.value) if hasattr(role, "value") else str(role)
                )
                input_messages.append(
                    InputMessage(
                        role=role_value,
                        parts=[Text(content=_safe_str(content))],
                    )
                )

        # Create LLM invocation with event_id as run_id
        llm_inv = LLMInvocation(
            request_model=_safe_str(model_name),
            input_messages=input_messages,
            attributes={},
            run_id=event_id,  # Use event_id as run_id for registry lookup
        )
        llm_inv.framework = "llamaindex"

        # Get the currently active span to establish parent-child relationship
        # First try to get from active agent context (workflow-based agents)
        parent_span = None
        if self._handler._agent_context_stack:
            # Get the current agent's span from the span registry
            _, agent_run_id = self._handler._agent_context_stack[-1]
            parent_span = self._handler._span_registry.get(agent_run_id)
        
        # Fallback to OpenTelemetry context if no agent span found
        if not parent_span:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                parent_span = current_span
        
        if parent_span:
            llm_inv.parent_span = parent_span

        # Start the LLM invocation (handler stores it in _entity_registry)
        self._handler.start_llm(llm_inv)

    def _handle_llm_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle LLM invocation end."""
        if not self._handler:
            return

        # Get the LLM invocation from handler's registry using event_id
        llm_inv = self._handler.get_entity(event_id)
        if not llm_inv or not isinstance(llm_inv, LLMInvocation):
            return

        if payload:
            # Extract response from payload
            response = payload.get("response")

            # Handle both dict and object types for response
            if response:
                # Get message - could be dict or object
                if isinstance(response, dict):
                    message = response.get("message", {})
                    raw_response = response.get("raw")
                else:
                    # response is a ChatResponse object
                    message = getattr(response, "message", None)
                    raw_response = getattr(response, "raw", None)

                # Extract content from message
                if message:
                    if isinstance(message, dict):
                        # Message is dict
                        blocks = message.get("blocks", [])
                        if blocks and isinstance(blocks[0], dict):
                            content = blocks[0].get("text", "")
                        else:
                            content = message.get("content", "")
                    else:
                        # Message is ChatMessage object
                        blocks = getattr(message, "blocks", [])
                        if blocks and len(blocks) > 0:
                            content = getattr(blocks[0], "text", "")
                        else:
                            content = getattr(message, "content", "")

                    # Create output message
                    llm_inv.output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=_safe_str(content))],
                            finish_reason="stop",
                        )
                    ]

                # Extract token usage from response.raw (OpenAI format)
                # LlamaIndex stores the raw API response (e.g., OpenAI response) in response.raw
                # raw_response could be a dict or an object (e.g., ChatCompletion from OpenAI)
                if raw_response:
                    # Try to get usage from dict or object
                    if isinstance(raw_response, dict):
                        usage = raw_response.get("usage", {})
                    else:
                        # It's an object, try to get usage attribute
                        usage = getattr(raw_response, "usage", None)
                    
                    if usage:
                        # usage could also be dict or object
                        if isinstance(usage, dict):
                            llm_inv.input_tokens = usage.get("prompt_tokens")
                            llm_inv.output_tokens = usage.get("completion_tokens")
                        else:
                            llm_inv.input_tokens = getattr(usage, "prompt_tokens", None)
                            llm_inv.output_tokens = getattr(usage, "completion_tokens", None)

        # Stop the LLM invocation
        self._handler.stop_llm(llm_inv)

    def _handle_embedding_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle embedding invocation start."""
        if not self._handler or not payload:
            return

        # Extract model information from payload
        serialized = payload.get("serialized", {})
        model_name = (
            serialized.get("model_name")
            or serialized.get("model")
            or "unknown"
        )

        # Detect provider from class name
        class_name = serialized.get("class_name", "")
        provider = detect_vendor_from_class(class_name)

        # Note: input texts are not available at start time in LlamaIndex
        # They will be available in the end event payload

        # Create embedding invocation with event_id as run_id
        emb_inv = EmbeddingInvocation(
            request_model=_safe_str(model_name),
            input_texts=[],  # Will be populated on end event
            provider=provider,
            attributes={},
            run_id=event_id,
        )
        emb_inv.framework = "llamaindex"

        # Get the currently active span to establish parent-child relationship
        # First try to get from active agent context (workflow-based agents)
        parent_span = None
        if self._handler._agent_context_stack:
            # Get the current agent's span from the span registry
            _, agent_run_id = self._handler._agent_context_stack[-1]
            parent_span = self._handler._span_registry.get(agent_run_id)
        
        # Fallback to OpenTelemetry context if no agent span found
        if not parent_span:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                parent_span = current_span
        
        if parent_span:
            emb_inv.parent_span = parent_span

        # Start the embedding invocation
        self._handler.start_embedding(emb_inv)

    def _handle_embedding_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle embedding invocation end."""
        if not self._handler:
            return

        # Get the embedding invocation from handler's registry using event_id
        emb_inv = self._handler.get_entity(event_id)
        if not emb_inv or not isinstance(emb_inv, EmbeddingInvocation):
            return

        if payload:
            # Extract input chunks (texts) from response
            # chunks is the list of input texts that were embedded
            chunks = payload.get("chunks", [])
            if chunks:
                emb_inv.input_texts = [_safe_str(chunk) for chunk in chunks]
            
            # Extract embedding vectors from response
            # embeddings is the list of output vectors
            embeddings = payload.get("embeddings", [])
            
            # Determine dimension from first embedding vector
            if embeddings and len(embeddings) > 0:
                first_embedding = embeddings[0]
                if isinstance(first_embedding, list):
                    emb_inv.dimension_count = len(first_embedding)
                elif hasattr(first_embedding, "__len__"):
                    emb_inv.dimension_count = len(first_embedding)

        # Stop the embedding invocation
        self._handler.stop_embedding(emb_inv)

    def _find_nearest_agent(self, parent_id: Optional[str]) -> Optional[AgentInvocation]:
        """Walk up parent chain to find the nearest agent invocation."""
        if not self._handler:
            return None
        current_id = parent_id
        while current_id:
            entity = self._handler.get_entity(current_id)
            if isinstance(entity, AgentInvocation):
                return entity
            # Move to parent
            current_id = getattr(entity, "parent_run_id", None)
            if current_id:
                current_id = str(current_id)
        return None

    def _handle_agent_step_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle agent step start - create AgentInvocation span."""
        if not self._handler or not payload:
            return

        # Extract agent information from payload
        task_id = payload.get("task_id", "")
        input_text = payload.get("input")
        step = payload.get("step")  # TaskStep object with agent metadata

        # Extract agent metadata from step or payload
        agent_name = None
        agent_type = None
        agent_description = None
        model_name = None
        
        if step and hasattr(step, "step_state"):
            # Try to get agent from step state
            step_state = step.step_state
            if hasattr(step_state, "agent"):
                agent = step_state.agent
                agent_name = getattr(agent, "name", None)
                agent_type = getattr(agent, "agent_type", None) or type(agent).__name__
                agent_description = getattr(agent, "description", None)
                # Try to get model from agent's LLM
                if hasattr(agent, "llm"):
                    llm = agent.llm
                    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)

        # Create AgentInvocation for the agent execution
        agent_invocation = AgentInvocation(
            name=f"agent.task.{task_id}" if task_id else "agent.invoke",
            run_id=event_id,
            parent_run_id=parent_id if parent_id else None,
            input_context=input_text if input_text else "",
            attributes={},
        )
        agent_invocation.framework = "llamaindex"
        
        # Set enhanced metadata
        if agent_name:
            agent_invocation.agent_name = _safe_str(agent_name)
        if agent_type:
            agent_invocation.agent_type = _safe_str(agent_type)
        if agent_description:
            agent_invocation.description = _safe_str(agent_description)
        if model_name:
            agent_invocation.model = _safe_str(model_name)
        
        self._handler.start_agent_invocation(agent_invocation)

    def _handle_agent_step_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle agent step end."""
        if not self._handler:
            return

        agent_invocation = self._handler.get_entity(event_id)
        if not agent_invocation or not isinstance(agent_invocation, AgentInvocation):
            return

        if payload:
            # Extract response/output if available
            response = payload.get("response")
            if response:
                agent_invocation.output_result = _safe_str(response)

        # Stop the agent invocation
        self._handler.stop_agent_invocation(agent_invocation)

    def _handle_function_call_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle function/tool call start."""
        if not self._handler or not payload:
            return

        # Extract tool information
        tool = payload.get("tool")
        if not tool:
            return
            
        tool_name = getattr(tool, "name", "unknown_tool") if hasattr(tool, "name") else "unknown_tool"
        tool_description = getattr(tool, "description", "") if hasattr(tool, "description") else ""
        
        # Extract function arguments
        function_call = payload.get("function_call", {})
        arguments = function_call if function_call else {}

        # Find nearest agent for context propagation
        context_agent = self._find_nearest_agent(parent_id) if parent_id else None

        # Create ToolCall entity
        tool_call = ToolCall(
            name=tool_name,
            arguments=arguments,
            id=event_id,
        )
        
        # Set attributes
        tool_call.attributes = {
            "tool.description": tool_description,
        }
        tool_call.run_id = event_id  # type: ignore[attr-defined]
        tool_call.parent_run_id = parent_id if parent_id else None  # type: ignore[attr-defined]
        tool_call.framework = "llamaindex"  # type: ignore[attr-defined]
        
        # Propagate agent context to tool call
        if context_agent:
            agent_name = getattr(context_agent, "agent_name", None) or getattr(context_agent, "name", None)
            if agent_name:
                tool_call.agent_name = _safe_str(agent_name)  # type: ignore[attr-defined]
            tool_call.agent_id = str(context_agent.run_id)  # type: ignore[attr-defined]
        
        # Start the tool call
        self._handler.start_tool_call(tool_call)

    def _handle_function_call_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle function/tool call end."""
        if not self._handler:
            return

        tool_call = self._handler.get_entity(event_id)
        if not tool_call or not isinstance(tool_call, ToolCall):
            return

        if payload:
            # Extract tool output/result
            tool_output = payload.get("tool_output")
            if tool_output:
                # Store the result as response
                tool_call.response = _safe_str(tool_output)  # type: ignore[attr-defined]

        # Stop the tool call
        self._handler.stop_tool_call(tool_call)
