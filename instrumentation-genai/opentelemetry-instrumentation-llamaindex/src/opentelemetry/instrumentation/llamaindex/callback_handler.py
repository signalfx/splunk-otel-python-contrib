from typing import Any, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    EmbeddingInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    RetrievalInvocation,
    Text,
    Workflow,
)

from .invocation_manager import _InvocationManager
from .vendor_detection import detect_vendor_from_class


def _safe_str(value: Any) -> str:
    """Safely convert value to string."""
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from dict-like or regular object.

    Uses try-except to support any dict-like object that implements .get(),
    not just dict instances. Falls back to getattr for regular objects.
    """
    try:
        return obj.get(key, default)
    except AttributeError:
        return getattr(obj, key, default)


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
        self._auto_workflow_ids: List[str] = []  # Track auto-created workflows (stack)
        self._invocation_manager = _InvocationManager()

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
        """Get parent span from invocation manager using parent_id."""
        if not parent_id:
            return None
        parent_entity = self._invocation_manager.get_invocation(parent_id)
        if parent_entity:
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

        # Extract model information and parameters from payload
        serialized = payload.get("serialized", {})
        model_name = (
            serialized.get("model") or serialized.get("model_name") or "unknown"
        )

        # Extract additional parameters if available
        temperature = serialized.get("temperature")
        max_tokens = serialized.get("max_tokens")
        top_p = serialized.get("top_p")
        frequency_penalty = serialized.get("frequency_penalty")
        presence_penalty = serialized.get("presence_penalty")
        stop = serialized.get("stop")
        seed = serialized.get("seed")

        # Extract messages from payload
        # LlamaIndex messages are ChatMessage objects with .content and .role properties
        messages = payload.get("messages", [])
        input_messages = []

        for msg in messages:
            # Handle ChatMessage objects (has .content property and .role attribute)
            if hasattr(msg, "content") and hasattr(msg, "role"):
                # Extract role - could be MessageRole enum
                role_value = _safe_str(
                    msg.role.value if hasattr(msg.role, "value") else msg.role
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

                role_value = _safe_str(role.value if hasattr(role, "value") else role)
                input_messages.append(
                    InputMessage(
                        role=role_value,
                        parts=[Text(content=_safe_str(content))],
                    )
                )

        # Create LLM invocation with all available parameters
        llm_inv = LLMInvocation(
            request_model=_safe_str(model_name),
            input_messages=input_messages,
            request_temperature=temperature,
            request_max_tokens=max_tokens,
            request_top_p=top_p,
            request_frequency_penalty=frequency_penalty,
            request_presence_penalty=presence_penalty,
            request_stop_sequences=stop if stop else [],
            request_seed=seed,
        )
        llm_inv.framework = "llamaindex"

        # Resolve parent_id to parent_span before starting, for proper span context
        parent_span = self._get_parent_span(parent_id)
        if parent_span:
            llm_inv.parent_span = parent_span  # type: ignore[attr-defined]
        # Start the LLM invocation
        llm_inv = self._handler.start_llm(llm_inv)

        # Store in invocation manager
        self._invocation_manager.add_invocation_state(
            event_id=event_id,
            parent_id=parent_id if parent_id else None,
            invocation=llm_inv,
        )

    def _handle_llm_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle LLM invocation end."""
        if not self._handler:
            return

        # Get the LLM invocation from invocation manager using event_id
        llm_inv = self._invocation_manager.get_invocation(event_id)
        if not llm_inv or not isinstance(llm_inv, LLMInvocation):
            return

        if payload:
            # Extract response from payload
            response = payload.get("response")

            # Handle both dict and object types for response
            if response:
                # Get message and raw_response - works for both dict and object
                message = _get_attr(response, "message")
                raw_response = _get_attr(response, "raw")

                # Extract content from message
                if message:
                    # Try to extract from blocks first (LlamaIndex format)
                    blocks = _get_attr(message, "blocks", [])
                    if blocks and len(blocks) > 0:
                        content = _get_attr(blocks[0], "text", "")
                    else:
                        # Fallback to direct content field
                        content = _get_attr(message, "content", "")

                    # Create output message
                    llm_inv.output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=_safe_str(content))],
                            finish_reason="stop",
                        )
                    ]

                # Extract token usage from raw_response
                if raw_response:
                    usage = _get_attr(raw_response, "usage")
                    if usage:
                        llm_inv.input_tokens = _get_attr(usage, "prompt_tokens")
                        llm_inv.output_tokens = _get_attr(usage, "completion_tokens")

        # Stop the LLM invocation
        llm_inv = self._handler.stop_llm(llm_inv)

        # Clean up from invocation manager if span is complete
        if not llm_inv.span.is_recording():
            self._invocation_manager.delete_invocation_state(event_id)

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

        # Create embedding invocation
        emb_inv = EmbeddingInvocation(
            request_model=_safe_str(model_name),
            input_texts=[],  # Will be populated on end event
        )
        emb_inv.framework = "llamaindex"
        if provider:
            emb_inv.provider = provider

        # Resolve parent_id to parent_span before starting, for proper span context
        parent_span = self._get_parent_span(parent_id)
        if parent_span:
            emb_inv.parent_span = parent_span  # type: ignore[attr-defined]
        # Start the embedding invocation
        emb_inv = self._handler.start_embedding(emb_inv)
        # Store in invocation manager
        self._invocation_manager.add_invocation_state(
            event_id=event_id,
            parent_id=parent_id if parent_id else None,
            invocation=emb_inv,
        )

    def _handle_embedding_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle embedding invocation end."""
        if not self._handler:
            return

        # Get the embedding invocation from invocation manager using event_id
        emb_inv = self._invocation_manager.get_invocation(event_id)
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
        emb_inv = self._handler.stop_embedding(emb_inv)

        # Clean up from invocation manager if span is complete
        if not emb_inv.span.is_recording():
            self._invocation_manager.delete_invocation_state(event_id)

    def _handle_query_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle query pipeline start - create Workflow if no parent exists."""
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
            workflow = self._handler.start_workflow(workflow)
            self._invocation_manager.add_invocation_state(
                event_id=event_id,
                parent_id=None,
                invocation=workflow,
            )

    def _handle_query_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle query pipeline end."""
        if not self._handler:
            return

        entity = self._invocation_manager.get_invocation(event_id)
        if not entity or not isinstance(entity, Workflow):
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
                entity.final_output = _safe_str(response_text)
        self._handler.stop_workflow(entity)
        self._invocation_manager.delete_invocation_state(event_id)

    def _handle_retrieve_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle retrieval start - create RetrievalInvocation for retrieve task.

        If no valid parent exists, automatically creates a root Workflow to hold
        the RAG operations. Sets proper parent-child span relationships.
        """
        if not self._handler or not payload:
            return

        query_str = payload.get("query_str", "")

        # If parent_id doesn't exist or doesn't resolve to a tracked entity,
        # check for a root LLM invocation; otherwise create a root Workflow.
        parent_entity = None
        parent_span = None
        parent_run_id = parent_id if parent_id else None

        if parent_id:
            parent_entity = self._invocation_manager.get_invocation(parent_id)
            if parent_entity:
                parent_span = getattr(parent_entity, "span", None)
            else:
                parent_run_id = None

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
            workflow = self._handler.start_workflow(workflow)
            # Track this auto-created workflow
            self._auto_workflow_ids.append(workflow_id)
            self._invocation_manager.add_invocation_state(
                event_id=workflow_id,
                parent_id=None,
                invocation=workflow,
            )
            parent_span = getattr(workflow, "span", None)
            parent_run_id = workflow_id

        # Create a retrieval invocation for the retrieval task
        retrieval = RetrievalInvocation(
            operation_name="retrieve",
            retriever_type="llamaindex_retriever",
            query=_safe_str(query_str),
            run_id=event_id,
            parent_run_id=parent_run_id,
            attributes={},
        )

        # Set parent_span if we have one
        if parent_span:
            retrieval.parent_span = parent_span  # type: ignore[attr-defined]

        retrieval = self._handler.start_retrieval(retrieval)
        self._invocation_manager.add_invocation_state(
            event_id=event_id,
            parent_id=parent_run_id,
            invocation=retrieval,
        )

    def _handle_retrieve_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle retrieval end - update RetrievalInvocation with retrieved nodes.

        Extracts document count, relevance scores, and document IDs from the
        retrieved nodes and stores them as attributes.
        """
        if not self._handler:
            return

        retrieval = self._invocation_manager.get_invocation(event_id)
        if not retrieval or not isinstance(retrieval, RetrievalInvocation):
            return

        if payload:
            nodes = payload.get("nodes", [])
            if nodes:
                # Set documents retrieved count
                retrieval.documents_retrieved = len(nodes)

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
        self._invocation_manager.delete_invocation_state(event_id)

    def _handle_synthesize_start(
        self,
        event_id: str,
        parent_id: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle synthesis start - no span needed, LLM invocation inside will be tracked."""
        # Synthesize events don't create their own span
        # The LLM invocation that happens during synthesis is already tracked
        pass

    def _handle_synthesize_end(
        self,
        event_id: str,
        payload: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Handle synthesis end - close auto-created workflow if present."""
        if not self._handler:
            return

        # If we auto-created a workflow, close the most recent one after synthesize completes
        # The workflow_id follows the pattern: {retrieve_event_id}_workflow
        if self._auto_workflow_ids:
            # Get the most recent auto-workflow (LIFO)
            workflow_id = self._auto_workflow_ids[-1]
            workflow = self._invocation_manager.get_invocation(workflow_id)
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
                self._invocation_manager.delete_invocation_state(workflow_id)
                # Remove from tracking after successful close
                self._auto_workflow_ids.remove(workflow_id)
