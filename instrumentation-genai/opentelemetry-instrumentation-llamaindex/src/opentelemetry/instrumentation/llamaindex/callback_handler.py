from typing import Any, Dict, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    RetrievalInvocation,
    Step,
    Text,
    Workflow,
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
        self._auto_workflow_id: Optional[str] = None  # Track auto-created workflow

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

    def _get_parent_span(self, parent_id: str) -> Optional[Any]:
        """Get parent span from handler's registry using parent_id."""
        if not self._handler or not parent_id:
            return None
        # Get the parent entity from handler's registry
        parent_entity = self._handler.get_entity(parent_id)
        if parent_entity:
            # Return the span attribute if it exists
            return getattr(parent_entity, "span", None)
        return None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Handle event start - processing LLM, EMBEDDING, QUERY, RETRIEVE, and SYNTHESIZE events."""
        if event_type == CBEventType.LLM:
            self._handle_llm_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.EMBEDDING:
            self._handle_embedding_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.QUERY:
            self._handle_query_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.RETRIEVE:
            self._handle_retrieve_start(event_id, parent_id, payload, **kwargs)
        elif event_type == CBEventType.SYNTHESIZE:
            self._handle_synthesize_start(event_id, parent_id, payload, **kwargs)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Handle event end - processing LLM, EMBEDDING, QUERY, RETRIEVE, and SYNTHESIZE events."""
        if event_type == CBEventType.LLM:
            self._handle_llm_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.EMBEDDING:
            self._handle_embedding_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.QUERY:
            self._handle_query_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.RETRIEVE:
            self._handle_retrieve_end(event_id, payload, **kwargs)
        elif event_type == CBEventType.SYNTHESIZE:
            self._handle_synthesize_end(event_id, payload, **kwargs)

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
            serialized.get("model") or serialized.get("model_name") or "unknown"
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
                    str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
                )
                # Extract content - this is a property that pulls from blocks[0].text
                content = _safe_str(msg.content)
                input_messages.append(
                    InputMessage(role=role_value, parts=[Text(content=content)])
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

                role_value = str(role.value) if hasattr(role, "value") else str(role)
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
            parent_run_id=parent_id if parent_id else None,  # Set parent for hierarchy
        )
        llm_inv.framework = "llamaindex"

        # Resolve parent_id to parent_span for proper span context
        parent_span = self._get_parent_span(parent_id)
        if parent_span:
            llm_inv.parent_span = parent_span  # type: ignore[attr-defined]

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
                            llm_inv.output_tokens = getattr(
                                usage, "completion_tokens", None
                            )

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
            serialized.get("model_name") or serialized.get("model") or "unknown"
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
            parent_run_id=parent_id if parent_id else None,  # Set parent for hierarchy
        )
        emb_inv.framework = "llamaindex"

        # Resolve parent_id to parent_span for proper span context
        parent_span = self._get_parent_span(parent_id)
        if parent_span:
            emb_inv.parent_span = parent_span  # type: ignore[attr-defined]

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

    def _handle_query_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle query pipeline start - create Workflow if no parent, else Step."""
        if not self._handler or not payload:
            return

        query_str = payload.get("query_str", "")

        # If no parent, this is the root workflow
        if not parent_id:
            workflow = Workflow(
                name="llama_index_query_pipeline",
                workflow_type="workflow",
                initial_input=_safe_str(query_str),
                attributes={},
                run_id=event_id,
            )
            workflow.framework = "llamaindex"
            self._handler.start_workflow(workflow)

    def _handle_query_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle query pipeline end."""
        if not self._handler:
            return

        entity = self._handler.get_entity(event_id)
        if not entity:
            return

        if isinstance(entity, Workflow):
            if payload:
                response = payload.get("response")
                if response:
                    # Extract response text
                    response_text = ""
                    if isinstance(response, dict):
                        response_text = response.get("response", "")
                    elif hasattr(response, "response"):
                        response_text = getattr(response, "response", "")
                    entity.final_output = _safe_str(response_text)
            self._handler.stop_workflow(entity)
        elif isinstance(entity, Step):
            if payload:
                response = payload.get("response")
                if response:
                    response_text = ""
                    if isinstance(response, dict):
                        response_text = response.get("response", "")
                    elif hasattr(response, "response"):
                        response_text = getattr(response, "response", "")
                    entity.output_data = _safe_str(response_text)
            self._handler.stop_step(entity)

    def _handle_retrieve_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retrieval start - create RetrievalInvocation for retrieve task."""
        if not self._handler or not payload:
            return

        query_str = payload.get("query_str", "")

        # If parent_id doesn't exist or doesn't resolve to a tracked entity,
        # create a root Workflow to hold the RAG operations
        parent_entity = self._handler.get_entity(parent_id) if parent_id else None

        if not parent_entity:
            # No valid parent - create auto-workflow
            workflow_id = f"{event_id}_workflow"
            workflow = Workflow(
                name="llama_index_rag",
                workflow_type="rag",
                initial_input=_safe_str(query_str),
                attributes={},
                run_id=workflow_id,
            )
            workflow.framework = "llamaindex"
            self._handler.start_workflow(workflow)
            # Track this auto-created workflow
            self._auto_workflow_id = workflow_id
            # Get the workflow's span to use as parent
            workflow_entity = self._handler.get_entity(workflow_id)
            if workflow_entity:
                parent_span = getattr(workflow_entity, "span", None)
            else:
                parent_span = None
        else:
            # Valid parent exists - resolve to parent_span
            parent_span = self._get_parent_span(parent_id)

        # Create a retrieval invocation for the retrieval task
        retrieval = RetrievalInvocation(
            operation_name="retrieve",
            retriever_type="llamaindex_retriever",
            query=_safe_str(query_str),
            run_id=event_id,
            parent_run_id=parent_id if parent_id else None,
            attributes={},
        )

        # Set parent_span if we have one
        if parent_span:
            retrieval.parent_span = parent_span  # type: ignore[attr-defined]

        self._handler.start_retrieval(retrieval)

    def _handle_retrieve_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle retrieval end - update RetrievalInvocation with retrieved nodes."""
        if not self._handler:
            return

        retrieval = self._handler.get_entity(event_id)
        if not retrieval or not isinstance(retrieval, RetrievalInvocation):
            return

        if payload:
            nodes = payload.get("nodes", [])
            if nodes:
                # Set document count
                retrieval.document_count = len(nodes)

                # Store scores and document IDs as attributes
                scores = []
                doc_ids = []
                for node in nodes:
                    if hasattr(node, "score") and node.score is not None:
                        scores.append(node.score)
                    if hasattr(node, "node_id"):
                        doc_ids.append(str(node.node_id))
                    elif hasattr(node, "id_"):
                        doc_ids.append(str(node.id_))

                if scores:
                    retrieval.attributes["retrieve.scores"] = scores
                if doc_ids:
                    retrieval.attributes["retrieve.document_ids"] = doc_ids

        self._handler.stop_retrieval(retrieval)

    def _handle_synthesize_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle synthesis start - create Step for synthesize task."""
        if not self._handler or not payload:
            return

        query_str = payload.get("query_str", "")

        # Create a step for the synthesis task
        step = Step(
            name="synthesize.task",
            step_type="synthesize",
            objective="Synthesize response from retrieved documents",
            input_data=_safe_str(query_str),
            run_id=event_id,
            parent_run_id=parent_id if parent_id else None,
            attributes={},
        )

        # Resolve parent_id to parent_span for proper span context
        parent_span = self._get_parent_span(parent_id)
        if parent_span:
            step.parent_span = parent_span  # type: ignore[attr-defined]

        self._handler.start_step(step)

    def _handle_synthesize_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle synthesis end - update step with synthesized response."""
        if not self._handler:
            return

        step = self._handler.get_entity(event_id)
        if not step or not isinstance(step, Step):
            return

        if payload:
            response = payload.get("response")
            if response:
                # Extract response text
                response_text = ""
                if isinstance(response, dict):
                    response_text = response.get("response", "")
                elif hasattr(response, "response"):
                    response_text = getattr(response, "response", "")
                step.output_data = _safe_str(response_text)

        self._handler.stop_step(step)

        # If we auto-created a workflow, close it after synthesize completes
        if self._auto_workflow_id:
            workflow = self._handler.get_entity(self._auto_workflow_id)
            if workflow and isinstance(workflow, Workflow):
                # Set final output from synthesize response
                if payload:
                    response = payload.get("response")
                    if response:
                        response_text = ""
                        if isinstance(response, dict):
                            response_text = response.get("response", "")
                        elif hasattr(response, "response"):
                            response_text = getattr(response, "response", "")
                        workflow.final_output = _safe_str(response_text)
                self._handler.stop_workflow(workflow)
                self._auto_workflow_id = None  # Reset for next query
