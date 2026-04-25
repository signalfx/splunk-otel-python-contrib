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

"""
Telemetry handler for GenAI invocations.

This module exposes the `TelemetryHandler` class, which manages the lifecycle of
GenAI (Generative AI) invocations and emits telemetry data (spans and related attributes).
It supports starting, stopping, and failing LLM invocations.

Classes:
    - TelemetryHandler: Manages GenAI invocation lifecycles and emits telemetry.

Functions:
    - get_telemetry_handler: Returns a singleton `TelemetryHandler` instance.

Usage:
    handler = get_telemetry_handler()

    # Create an invocation object with your request data
    invocation = LLMInvocation(
        request_model="my-model",
        input_messages=[...],
        provider="my-provider",
        attributes={"custom": "attr"},
    )

    # Start the invocation (opens a span)
    handler.start_llm(invocation)

    # Populate outputs and any additional attributes, then stop (closes the span)
    invocation.output_messages = [...]
    invocation.attributes.update({"more": "attrs"})
    handler.stop_llm(invocation)

    # Or, in case of error
    # handler.fail_llm(invocation, Error(type="...", message="..."))
"""

import logging
import os
import threading
import timeit
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

try:
    from opentelemetry.util.genai.debug import genai_debug_log
except Exception:  # pragma: no cover - fallback if debug module missing

    def genai_debug_log(*_args: Any, **_kwargs: Any) -> None:  # type: ignore
        return None


from opentelemetry import _events as _otel_events
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry._logs import Logger, LoggerProvider, get_logger
from opentelemetry.metrics import MeterProvider, get_meter
from opentelemetry.sdk.trace.sampling import Decision, TraceIdRatioBased
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import (
    Span,
    TracerProvider,
    get_tracer,
)
from opentelemetry.util.genai._embedding_invocation import (
    EmbeddingInvocation as NewEmbeddingInvocation,
)
from opentelemetry.util.genai._inference_invocation import (
    InferenceInvocation,
)

# New-style invocation classes
from opentelemetry.util.genai._invocation import (
    GenAIInvocation,
)
from opentelemetry.util.genai._tool_invocation import (
    ToolInvocation,
)
from opentelemetry.util.genai._workflow_invocation import (
    WorkflowInvocation,
)
from opentelemetry.util.genai.emitters.configuration import (
    build_emitter_pipeline,
)
from opentelemetry.util.genai.span_context import (
    extract_span_context,
    span_context_hex_ids,
    store_span_context,
)
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    ContentCapturingMode,
    EmbeddingInvocation,
    Error,
    EvaluationResult,
    GenAI,
    InputMessage,
    LLMInvocation,
    MCPOperation,
    RetrievalInvocation,
    Step,
    ToolCall,
    Workflow,
)
from opentelemetry.util.genai.utils import (
    get_content_capturing_mode,
    is_truthy_env,
    load_completion_callbacks,
    parse_callback_filter,
)
from opentelemetry.util.genai.version import __version__

from .callbacks import CompletionCallback
from .config import parse_env
from .environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS,
    OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC,
    OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW,
)

_LOGGER = logging.getLogger(__name__)

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


@dataclass
class GenAIContext:
    """Holds conversation context and association properties for GenAI operations.

    This dataclass stores a conversation identifier and arbitrary key-value
    association properties that are automatically propagated to all nested
    GenAI operations (LLM calls, agent invocations, tool calls, etc.).

    Association properties are emitted on spans as
    ``gen_ai.association.properties.<key>``.

    Attributes:
        conversation_id: Unique identifier for the conversation
            (propagated as ``gen_ai.conversation.id``).
        properties: Arbitrary key-value association properties
            (e.g. ``{"user.id": "alice", "customer.id": "acme"}``).
    """

    conversation_id: Optional[str] = None
    properties: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if no context values are set."""
        return self.conversation_id is None and not self.properties


# Module-level context variable for GenAI context propagation
_genai_context: ContextVar[GenAIContext] = ContextVar(
    "genai_context", default=GenAIContext()
)

# Tracks the most recently started GenAI span in the current context.
# Used as a fallback parent for instrumentations that don't explicitly
# set parent_span (e.g. sync frameworks like CrewAI / FastMCP).
# Uses ContextVar.set() (never reset()) to avoid cross-task token errors.
_current_genai_span: ContextVar[Optional[Span]] = ContextVar(
    "_current_genai_span", default=None
)


def set_genai_context(
    conversation_id: Optional[str] = None,
    properties: Optional[dict[str, Any]] = None,
) -> None:
    """Set GenAI context that propagates to all nested GenAI operations.

    This function sets conversation context and association properties that
    will be automatically applied to all GenAI invocations (LLM calls,
    agent invocations, tool calls, etc.) started within the current context.
    The context is thread-safe and async-safe using Python's contextvars.

    Association properties are emitted on spans as
    ``gen_ai.association.properties.<key>``.

    Args:
        conversation_id: Unique identifier for the conversation
            (emitted as ``gen_ai.conversation.id``).
        properties: Arbitrary key-value association properties
            (e.g. ``{"user.id": "alice", "customer.id": "acme"}``).

    Example:
        >>> from opentelemetry.util.genai import set_genai_context
        >>> set_genai_context(
        ...     conversation_id="conv-123",
        ...     properties={"user.id": "alice", "customer.id": "acme"},
        ... )
        >>> # All subsequent GenAI operations will have these attributes
        >>> result = chain.invoke({"input": "Hello"})
    """
    ctx = GenAIContext(
        conversation_id=conversation_id,
        properties=dict(properties) if properties else {},
    )
    _genai_context.set(ctx)


def get_genai_context() -> GenAIContext:
    """Get the current GenAI context.

    Returns:
        The current GenAIContext, or an empty GenAIContext if none is set.

    Example:
        >>> ctx = get_genai_context()
        >>> print(f"Conversation: {ctx.conversation_id}, Props: {ctx.properties}")
    """
    return _genai_context.get()


def clear_genai_context() -> None:
    """Clear the current GenAI context.

    Resets the GenAI context to an empty state. Useful for cleanup
    after processing a request.

    Example:
        >>> set_genai_context(conversation_id="conv-123")
        >>> # ... process request ...
        >>> clear_genai_context()
    """
    _genai_context.set(GenAIContext())


@contextmanager
def genai_context(
    conversation_id: Optional[str] = None,
    properties: Optional[dict[str, Any]] = None,
) -> Iterator[GenAIContext]:
    """Context manager for GenAI context that auto-restores on exit.

    This is the recommended way to set GenAI context for a request or
    operation, as it automatically restores the previous context on exit.

    Association properties are emitted on spans as
    ``gen_ai.association.properties.<key>``.

    Args:
        conversation_id: Unique identifier for the conversation
            (emitted as ``gen_ai.conversation.id``).
        properties: Arbitrary key-value association properties
            (e.g. ``{"user.id": "alice", "customer.id": "acme"}``).

    Yields:
        The GenAIContext object for the duration of the context.

    Example:
        >>> from opentelemetry.util.genai import genai_context
        >>> with genai_context(
        ...     conversation_id="conv-123",
        ...     properties={"user.id": "alice"},
        ... ):
        ...     # All GenAI operations here will have context attributes
        ...     result = chain.invoke({"input": "Hello"})
        >>> # Context is automatically restored after exiting
    """
    ctx = GenAIContext(
        conversation_id=conversation_id,
        properties=dict(properties) if properties else {},
    )
    token = _genai_context.set(ctx)

    try:
        yield ctx
    finally:
        _genai_context.reset(token)


def _apply_genai_context(invocation: GenAI) -> None:
    """Apply GenAI context to an invocation if not already set.

    Internal helper that applies the current GenAI context to a GenAI
    invocation object. Can be disabled via the
    ``OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION`` environment variable
    (default: ``true``).

    Priority order:

    1. Explicit value set on invocation object
    2. Value from contextvars GenAI context

    Association properties from the context are merged into the invocation's
    ``association_properties`` dict. Invocation-level properties take
    priority over context-level properties for the same key.

    Args:
        invocation: The GenAI invocation to apply context to.
    """
    from .environment_variables import (
        OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION,
    )

    # Check if context propagation is disabled
    prop_val = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CONTEXT_PROPAGATION, "true"
    )
    if prop_val.strip().lower() not in _TRUTHY_VALUES:
        return

    ctx = _genai_context.get()

    # Apply conversation_id: invocation > contextvars
    if not invocation.conversation_id:
        if ctx.conversation_id:
            invocation.conversation_id = ctx.conversation_id

    # Merge association properties: context values, then invocation overrides
    if ctx.properties:
        merged = dict(ctx.properties)
        merged.update(invocation.association_properties)
        invocation.association_properties = merged


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events. Evaluation execution & emission is
    delegated to EvaluationManager for extensibility (mirrors emitter design).

    This class is a **singleton**: all calls to ``TelemetryHandler(...)`` and
    ``get_telemetry_handler(...)`` return the same process-wide instance so
    that handler-internal context stacks (workflow, agent) are shared across
    instrumentation boundaries.

    Use ``TelemetryHandler._reset_for_testing()`` in test teardown to clear
    the singleton.
    """

    _instance: Optional["TelemetryHandler"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        tracer_provider: TracerProvider | None = None,
        logger_provider: LoggerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> "TelemetryHandler":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            instance = super().__new__(cls)
            cls._instance = instance
            return instance

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Reset the singleton instance.

        Intended **only** for test teardown.  Clears the singleton so the next
        call to ``TelemetryHandler(...)`` or ``get_telemetry_handler(...)``
        creates a fresh instance.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
            cls._instance = None
            if hasattr(get_telemetry_handler, "_default_handler"):
                delattr(get_telemetry_handler, "_default_handler")

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        logger_provider: LoggerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ):
        if getattr(self, "_initialized", False):
            supplied = {
                name
                for name, val in (
                    ("tracer_provider", tracer_provider),
                    ("logger_provider", logger_provider),
                    ("meter_provider", meter_provider),
                )
                if val is not None
            }
            if supplied:
                _LOGGER.warning(
                    "TelemetryHandler is a singleton and has already been "
                    "initialized; the following provider arguments are being "
                    "ignored: %s. To use different providers, call "
                    "TelemetryHandler._reset_for_testing() first (test only).",
                    ", ".join(sorted(supplied)),
                )
            return
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_37_0.value,
        )

        # Logger for content events (uses Logs API, not Events API)
        self._content_logger: Logger = get_logger(
            __name__,
            __version__,
            logger_provider=logger_provider,
            schema_url=Schemas.V1_37_0.value,
        )
        self._meter_provider = meter_provider
        meter = get_meter(
            __name__,
            __version__,
            meter_provider=meter_provider,
            schema_url=Schemas.V1_37_0.value,
        )

        self._event_logger = _otel_events.get_event_logger(__name__)

        settings = parse_env()

        evaluation_sample_rate = settings.evaluation_sample_rate
        self._sampler = TraceIdRatioBased(evaluation_sample_rate)

        # Check if single metric mode is enabled (default: True)
        env_value = os.getenv(
            OTEL_INSTRUMENTATION_GENAI_EVALS_USE_SINGLE_METRIC
        )
        if env_value is None:
            use_single_metric = True  # Default to single metric mode
        else:
            use_single_metric = is_truthy_env(env_value)

        _CANONICAL_METRICS = {
            "relevance",
            "hallucination",
            "sentiment",
            "toxicity",
            "bias",
        }

        # Bucket boundaries for evaluation metrics (0-1 score range)
        # Appropriate for DeepEval and other evaluation frameworks that return scores in [0, 1]
        _GEN_AI_EVALUATION_SCORE_BUCKETS = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

        if use_single_metric:
            # Single evaluation histogram for all evaluation types:
            # gen_ai.evaluation.score (with gen_ai.evaluation.name attribute)
            self._evaluation_histogram: Any = None

            def _get_eval_histogram(canonical_name: str):
                name = canonical_name.strip().lower()
                if name not in _CANONICAL_METRICS:
                    return None  # ignore unknown metrics (no emission)
                # Return the same histogram for all metrics
                if self._evaluation_histogram is None:
                    try:
                        self._evaluation_histogram = meter.create_histogram(
                            name="gen_ai.evaluation.score",
                            unit="1",
                            description="GenAI evaluation score (0-1 where applicable), distinguished by gen_ai.evaluation.name attribute",
                            explicit_bucket_boundaries_advisory=_GEN_AI_EVALUATION_SCORE_BUCKETS,
                        )
                    except Exception:  # pragma: no cover - defensive
                        return None
                return self._evaluation_histogram
        else:
            # Multiple evaluation histograms (legacy behavior):
            # gen_ai.evaluation.(relevance|hallucination|sentiment|toxicity|bias)
            self._evaluation_histograms: dict[str, Any] = {}

            def _get_eval_histogram(canonical_name: str):
                name = canonical_name.strip().lower()
                if name not in _CANONICAL_METRICS:
                    return None  # ignore unknown metrics (no emission)
                full_name = f"gen_ai.evaluation.{name}"
                hist = self._evaluation_histograms.get(full_name)
                if hist is not None:
                    return hist
                try:
                    hist = meter.create_histogram(
                        name=full_name,
                        unit="1",
                        description=f"GenAI evaluation metric '{name}' (0-1 score where applicable)",
                        explicit_bucket_boundaries_advisory=_GEN_AI_EVALUATION_SCORE_BUCKETS,
                    )
                    self._evaluation_histograms[full_name] = hist
                except Exception:  # pragma: no cover - defensive
                    return None
                return hist

        self._get_eval_histogram = _get_eval_histogram  # type: ignore[attr-defined]

        self._completion_callbacks: list[CompletionCallback] = []
        composite, capture_control = build_emitter_pipeline(
            tracer=self._tracer,
            meter=meter,
            event_logger=self._event_logger,
            content_logger=self._content_logger,
            evaluation_histogram=self._get_eval_histogram,
            settings=settings,
        )
        self._emitter = composite
        self._capture_control = capture_control
        self._evaluation_manager = None
        # Active agent identity stack (name, id) for implicit propagation to nested operations
        # agent_id may be None if not provided by instrumentation
        self._agent_context_stack: list[tuple[str, Optional[str]]] = []
        self._initialize_default_callbacks()
        self._initialized = True

    def _should_sample_for_evaluation(self, trace_id: Optional[int]) -> bool:
        try:
            if trace_id:
                sampling_result = self._sampler.should_sample(
                    trace_id=trace_id,
                    parent_context=None,
                    name="",
                )
                if (
                    sampling_result
                    and sampling_result.decision is Decision.RECORD_AND_SAMPLE
                ):
                    return True
                else:
                    return False
            else:  # TODO remove else branch when trace_id is set on all invocations
                _LOGGER.debug(
                    "Trace based sampling not applied as trace id is not set.",
                    exc_info=True,
                )
                return True
        except Exception:
            _LOGGER.debug("Sampler raised an exception", exc_info=True)
            return True

    def _refresh_capture_content(
        self,
    ):  # re-evaluate env each start in case singleton created before patching
        try:
            mode = get_content_capturing_mode()
            emitters = list(
                self._emitter.iter_emitters(("span", "content_events"))
            )
            # Determine new values for span-like emitters
            new_value_span = mode in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
            control = getattr(self, "_capture_control", None)
            span_capture_allowed = True
            if control is not None:
                span_capture_allowed = control.span_allowed
            if is_truthy_env(
                os.environ.get(
                    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
                )
            ):
                span_capture_allowed = True
            # Respect the content capture mode for all generator kinds
            new_value_events = mode in (
                ContentCapturingMode.EVENT_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
            for em in emitters:
                role = getattr(em, "role", None)
                if role == "content_event" and hasattr(em, "_capture_content"):
                    try:
                        em._capture_content = new_value_events  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif role in ("span", "traceloop_compat") and hasattr(
                    em, "set_capture_content"
                ):
                    try:
                        desired_span = new_value_span and span_capture_allowed
                        if role == "traceloop_compat":
                            desired = desired_span or new_value_events
                        else:
                            desired = desired_span
                        em.set_capture_content(desired)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # Propagate tool-definitions flag to span emitters
                if hasattr(em, "set_capture_tool_definitions"):
                    try:
                        em.set_capture_tool_definitions(  # type: ignore[attr-defined]
                            is_truthy_env(
                                os.environ.get(
                                    OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS
                                )
                            )
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    # -- Span parenting helpers -----------------------------------------------
    # These maintain a ContextVar-based "current GenAI span" so that sync
    # instrumentations (CrewAI, FastMCP) get correct parent-child links
    # without needing to set parent_span themselves.  Async instrumentations
    # (LangChain, OpenAI Agents) already set parent_span explicitly; the
    # ContextVar fallback is simply ignored for them.

    @staticmethod
    def _inherit_parent_span(invocation: GenAI) -> None:
        """If parent_span is unset, inherit from the current GenAI span."""
        if getattr(invocation, "parent_span", None) is None:
            current = _current_genai_span.get()
            if current is not None:
                invocation.parent_span = current

    @staticmethod
    def _push_current_span(invocation: GenAI) -> None:
        """After span creation, track this span as current for child resolution.

        Also attach to OTel context so downstream instrumentations (HTTP, DB,
        MCP transport) see the correct parent.  ``_pop_current_span`` handles
        detach failures gracefully for cross-task / ``copy_context`` scenarios.
        """
        span = getattr(invocation, "span", None)
        if span is not None:
            _current_genai_span.set(span)
            ctx = trace.set_span_in_context(span)
            invocation._otel_context_token = context_api.attach(ctx)  # type: ignore[attr-defined]

    @staticmethod
    def _pop_current_span(invocation: GenAI) -> None:
        """After completion, restore current span to parent (effectively a pop).

        The detach may fail with ValueError when a framework (e.g. LangGraph)
        executes callbacks inside ``copy_context().run()`` boundaries, causing
        the token to belong to a different context copy.  This is harmless --
        the GenAI parent-child hierarchy is managed by ``_current_genai_span``
        and is unaffected.  We call the runtime context directly to avoid
        the noisy ERROR log that ``context_api.detach`` emits on failure.
        """
        _current_genai_span.set(getattr(invocation, "parent_span", None))
        token = getattr(invocation, "_otel_context_token", None)
        if token is not None:
            try:
                context_api._RUNTIME_CONTEXT.detach(token)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    # -- Internal helpers ----------------------------------------------------

    def _invocation_components(self) -> dict:
        """Return the component dict needed by new-style invocation classes."""
        return dict(
            emitter=self._emitter,
            agent_context_stack=self._agent_context_stack,
            completion_callbacks=self._completion_callbacks,
            sampler_fn=self._should_sample_for_evaluation,
            meter_provider=getattr(self, "_meter_provider", None),
            capture_refresh_fn=self._refresh_capture_content,
        )

    def _maybe_force_flush(self) -> None:
        """Force flush metrics if a custom meter provider is available."""
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover - defensive
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass

    # -- New-style factory methods -------------------------------------------
    # These construct and start new invocation objects in one call.
    # The returned invocation manages its own lifecycle via stop() / fail().

    def start_inference(
        self,
        provider: str,
        *,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> InferenceInvocation:
        """Create and start an LLM inference invocation.

        Set remaining attributes (input_messages, temperature, etc.) on the
        returned invocation, then call ``invocation.stop()`` or
        ``invocation.fail()``.
        """
        return InferenceInvocation(
            **self._invocation_components(),
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )

    def start_new_embedding(
        self,
        provider: str,
        *,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> NewEmbeddingInvocation:
        """Create and start an embedding invocation.

        Set remaining attributes (encoding_formats, etc.) on the returned
        invocation, then call ``invocation.stop()`` or ``invocation.fail()``.
        """
        return NewEmbeddingInvocation(
            **self._invocation_components(),
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )

    def start_tool(
        self,
        name: str,
        *,
        arguments: Any = None,
        tool_call_id: Optional[str] = None,
        tool_type: Optional[str] = None,
        tool_description: Optional[str] = None,
    ) -> ToolInvocation:
        """Create and start a tool invocation.

        Set tool_result on the returned invocation when done, then call
        ``invocation.stop()`` or ``invocation.fail()``.
        """
        return ToolInvocation(
            **self._invocation_components(),
            name=name,
            arguments=arguments,
            tool_id=tool_call_id,
            tool_type=tool_type,
            tool_description=tool_description,
        )

    def start_new_workflow(
        self,
        *,
        name: Optional[str] = None,
        workflow_type: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
    ) -> WorkflowInvocation:
        """Create and start a workflow invocation.

        Set remaining attributes on the returned invocation, then call
        ``invocation.stop()`` or ``invocation.fail()``.
        """
        return WorkflowInvocation(
            **self._invocation_components(),
            name=name or "",
            workflow_type=workflow_type,
            framework=framework,
            system=system,
        )

    # -- Context manager convenience methods ---------------------------------

    def inference(
        self,
        provider: str,
        *,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
    ):
        """Context manager for LLM inference invocations.

        Starts the span on entry. On normal exit, finalizes the invocation
        and ends the span. If an exception occurs, marks the span as error,
        ends it, and re-raises the original exception.
        """
        return self.start_inference(
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )._managed()

    def embedding(
        self,
        provider: str,
        *,
        request_model: Optional[str] = None,
        server_address: Optional[str] = None,
        server_port: Optional[int] = None,
    ):
        """Context manager for embedding invocations."""
        return self.start_new_embedding(
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )._managed()

    def tool(
        self,
        name: str,
        *,
        arguments: Any = None,
        tool_call_id: Optional[str] = None,
        tool_type: Optional[str] = None,
        tool_description: Optional[str] = None,
    ):
        """Context manager for tool invocations."""
        return self.start_tool(
            name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            tool_type=tool_type,
            tool_description=tool_description,
        )._managed()

    def new_workflow(
        self,
        name: Optional[str] = None,
        *,
        workflow_type: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
    ):
        """Context manager for workflow invocations."""
        return self.start_new_workflow(
            name=name,
            workflow_type=workflow_type,
            framework=framework,
            system=system,
        )._managed()

    # -- Deprecated lifecycle methods ----------------------------------------
    # The following methods are preserved for backward compatibility.
    # Prefer the new factory methods (start_inference, start_tool, etc.)
    # and invocation.stop() / invocation.fail() instead.

    def start_llm(
        self,
        invocation: LLMInvocation,
    ) -> LLMInvocation:
        """Start an LLM invocation and create a pending span entry."""
        invocation._handler = self
        # Ensure capture content settings are current
        self._refresh_capture_content()
        genai_debug_log("handler.start_llm.begin", invocation)
        # Apply GenAI context from contextvars if not already set
        _apply_genai_context(invocation)
        # Implicit agent inheritance
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        self._inherit_parent_span(invocation)
        self._emitter.on_start(invocation)
        self._push_current_span(invocation)
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            if trace_hex and span_hex:
                genai_debug_log(
                    "handler.start_llm.span_created",
                    invocation,
                    trace_id=trace_hex,
                    span_id=span_hex,
                )
            else:
                genai_debug_log("handler.start_llm.no_span", invocation)
        except Exception:  # pragma: no cover
            pass
        return invocation

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Finalize an LLM invocation successfully and end its span."""
        invocation.end_time = timeit.default_timer()

        # Determine if this invocation should be sampled for evaluation
        invocation.sample_for_evaluation = self._should_sample_for_evaluation(
            invocation.trace_id
        )

        self._notify_completion(invocation)
        self._emitter.on_end(invocation)
        self._pop_current_span(invocation)
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            genai_debug_log(
                "handler.stop_llm.complete",
                invocation,
                duration_ms=round(
                    (invocation.end_time - invocation.start_time) * 1000, 3
                )
                if invocation.end_time
                else None,
                trace_id=trace_hex,
                span_id=span_hex,
            )
        except Exception:  # pragma: no cover
            pass
        self._maybe_force_flush()
        return invocation

    def fail_llm(
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        invocation.end_time = timeit.default_timer()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._pop_current_span(invocation)
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            genai_debug_log(
                "handler.fail_llm.error",
                invocation,
                error_type=getattr(error, "type", None),
                error_message=getattr(error, "message", None),
                trace_id=trace_hex,
                span_id=span_hex,
            )
        except Exception:  # pragma: no cover
            pass
        self._maybe_force_flush()
        return invocation

    def start_embedding(
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Start an embedding invocation and create a pending span entry."""
        invocation._handler = self
        self._refresh_capture_content()
        # Apply GenAI context from contextvars if not already set
        _apply_genai_context(invocation)
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        invocation.start_time = timeit.default_timer()
        self._inherit_parent_span(invocation)
        self._emitter.on_start(invocation)
        self._push_current_span(invocation)
        return invocation

    def stop_embedding(
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Finalize an embedding invocation successfully and end its span."""
        invocation.end_time = timeit.default_timer()

        invocation.sample_for_evaluation = self._should_sample_for_evaluation(
            invocation.trace_id
        )

        self._notify_completion(invocation)
        self._emitter.on_end(invocation)
        self._pop_current_span(invocation)
        self._maybe_force_flush()
        return invocation

    def fail_embedding(
        self, invocation: EmbeddingInvocation, error: Error
    ) -> EmbeddingInvocation:
        """Fail an embedding invocation and end its span with error status."""
        invocation.end_time = timeit.default_timer()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._pop_current_span(invocation)
        self._maybe_force_flush()
        return invocation

    def start_retrieval(
        self, invocation: RetrievalInvocation
    ) -> RetrievalInvocation:
        """Start a retrieval invocation and create a pending span entry."""
        invocation._handler = self
        self._refresh_capture_content()
        # Apply GenAI context from contextvars if not already set
        _apply_genai_context(invocation)
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        invocation.start_time = timeit.default_timer()
        self._inherit_parent_span(invocation)
        self._emitter.on_start(invocation)
        self._push_current_span(invocation)
        return invocation

    def stop_retrieval(
        self, invocation: RetrievalInvocation
    ) -> RetrievalInvocation:
        """Finalize a retrieval invocation successfully and end its span."""
        invocation.end_time = timeit.default_timer()

        invocation.sample_for_evaluation = self._should_sample_for_evaluation(
            invocation.trace_id
        )

        self._emitter.on_end(invocation)
        self._pop_current_span(invocation)
        self._notify_completion(invocation)
        # Force flush metrics if a custom provider with force_flush is present
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    def fail_retrieval(
        self, invocation: RetrievalInvocation, error: Error
    ) -> RetrievalInvocation:
        """Fail a retrieval invocation and end its span with error status."""
        invocation.end_time = timeit.default_timer()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._pop_current_span(invocation)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    # ToolCall lifecycle --------------------------------------------------
    def start_tool_call(self, invocation: ToolCall) -> ToolCall:
        """Start a tool call invocation and create a pending span entry."""
        invocation._handler = self
        _apply_genai_context(invocation)
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        self._inherit_parent_span(invocation)
        self._emitter.on_start(invocation)
        self._push_current_span(invocation)
        return invocation

    def stop_tool_call(self, invocation: ToolCall) -> ToolCall:
        """Finalize a tool call invocation successfully and end its span."""
        invocation.end_time = timeit.default_timer()

        invocation.sample_for_evaluation = self._should_sample_for_evaluation(
            invocation.trace_id
        )

        self._notify_completion(invocation)
        self._emitter.on_end(invocation)
        self._pop_current_span(invocation)
        return invocation

    def fail_tool_call(self, invocation: ToolCall, error: Error) -> ToolCall:
        """Fail a tool call invocation and end its span with error status."""
        invocation.end_time = timeit.default_timer()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._pop_current_span(invocation)
        return invocation

    # MCPOperation lifecycle (non-tool-call MCP operations) ----------------
    def start_mcp_operation(self, op: MCPOperation) -> MCPOperation:
        """Start a non-tool-call MCP operation (list, read, get, etc.)."""
        _apply_genai_context(op)
        if (
            not op.agent_name or not op.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not op.agent_name:
                op.agent_name = top_name
            if not op.agent_id:
                op.agent_id = top_id
        self._inherit_parent_span(op)
        self._emitter.on_start(op)
        self._push_current_span(op)
        return op

    def stop_mcp_operation(self, op: MCPOperation) -> MCPOperation:
        """Finalize a non-tool-call MCP operation successfully."""
        op.end_time = timeit.default_timer()
        self._emitter.on_end(op)
        self._pop_current_span(op)
        return op

    def fail_mcp_operation(
        self, op: MCPOperation, error: Error
    ) -> MCPOperation:
        """Fail a non-tool-call MCP operation."""
        op.end_time = timeit.default_timer()
        self._emitter.on_error(error, op)
        self._pop_current_span(op)
        return op

    @staticmethod
    def _maybe_mark_conversation_root(entity: GenAI) -> None:
        """Auto-mark entity as conversation root when no parent span exists.

        Only Workflow and AgentInvocation are eligible — other entity types
        (LLMInvocation, Step, ToolCall, AgentCreation, etc.) are never roots.

        If the instrumentation has not explicitly set conversation_root and
        the entity has no parent_span, this is the invocation-level root.
        """
        if not isinstance(entity, (Workflow, AgentInvocation)):
            return
        if entity.conversation_root is None and not getattr(
            entity, "parent_span", None
        ):
            entity.conversation_root = True

    def should_use_workflow_root(
        self,
        force_workflow: bool = False,
        workflow_name: Optional[str] = None,
    ) -> bool:
        """Decide if root entity should be Workflow (vs AgentInvocation).

        Use this to determine root entity type before creating it, especially
        when the instrumentation needs to set type-specific fields before
        starting the entity.

        Returns True when:
        - force_workflow is True, or
        - workflow_name is a non-empty string, or
        - OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW env var is truthy.

        Example:
            if handler.should_use_workflow_root(workflow_name=override):
                wf = Workflow(name=override or default, ...)
                handler.start_workflow(wf)
            else:
                agent = AgentInvocation(...)
                agent.agent_name = ...  # set before start
                handler.start_agent(agent)
        """
        if force_workflow or workflow_name:
            return True
        return is_truthy_env(
            os.environ.get(
                OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW, ""
            )
        )

    def create_and_start_root(
        self,
        name: str,
        *,
        force_workflow: bool = False,
        workflow_name: Optional[str] = None,
        workflow_type: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        input_messages: Optional[list[InputMessage]] = None,
        conversation_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Workflow | AgentInvocation:
        """Create and start a root entity (Workflow or AgentInvocation).

        By default creates AgentInvocation. Creates Workflow when:
        - force_workflow is True, or
        - workflow_name is a non-empty string, or
        - OTEL_INSTRUMENTATION_GENAI_ROOT_SPAN_AS_WORKFLOW env var is truthy.

        Use the generic lifecycle methods to complete the entity:
        - ``finish(entity)`` to end successfully
        - ``fail(entity, error)`` to end with error

        Args:
            name: Entity name (agent name or default workflow name).
            force_workflow: If True, creates Workflow regardless of env var.
            workflow_name: Optional workflow name. If provided, creates
                Workflow and uses this as the span name.
            workflow_type: Workflow type (e.g., "crewai.crew"). Ignored for
                AgentInvocation.
            framework: Framework name (e.g., "crewai", "langchain").
            system: System name (e.g., "crewai", "langgraph").
            input_messages: Input messages to attach.
            conversation_id: Conversation ID to set.
            attributes: Additional span attributes.

        Returns:
            The created and started Workflow or AgentInvocation.
        """
        use_workflow = self.should_use_workflow_root(
            force_workflow, workflow_name
        )
        attrs = attributes or {}

        if use_workflow:
            wf_name = workflow_name or name
            entity: Workflow | AgentInvocation = Workflow(
                name=wf_name,
                workflow_type=workflow_type,
                framework=framework,
                system=system,
                attributes=attrs,
            )
            if input_messages:
                entity.input_messages = input_messages
            if conversation_id:
                entity.conversation_id = conversation_id
            self.start_workflow(entity)
        else:
            entity = AgentInvocation(
                name=name,
                framework=framework,
                system=system,
                attributes=attrs,
            )
            entity.agent_name = name
            if input_messages:
                entity.input_messages = input_messages
            if conversation_id:
                entity.conversation_id = conversation_id
            self.start_agent(entity)

        return entity

    # Workflow lifecycle --------------------------------------------------
    def start_workflow(self, workflow: Workflow) -> Workflow:
        """Start a workflow and create a pending span entry."""
        workflow._handler = self
        self._refresh_capture_content()
        _apply_genai_context(workflow)
        self._inherit_parent_span(workflow)
        self._maybe_mark_conversation_root(workflow)
        self._emitter.on_start(workflow)
        self._push_current_span(workflow)
        return workflow

    def _handle_evaluation_results(
        self, invocation: GenAI, results: list[EvaluationResult]
    ) -> None:
        if not results:
            return
        try:
            self._emitter.on_evaluation_results(results, invocation)
        except Exception:  # pragma: no cover - defensive
            pass

    def evaluation_results(
        self, invocation: GenAI, results: list[EvaluationResult]
    ) -> None:
        """Public hook for completion callbacks to report evaluation output."""

        try:
            genai_debug_log(
                "handler.evaluation_results.begin",
                invocation,
                result_count=len(results),
            )
        except Exception:  # pragma: no cover - defensive
            pass
        self._handle_evaluation_results(invocation, results)
        try:
            genai_debug_log(
                "handler.evaluation_results.end",
                invocation,
                result_count=len(results),
            )
        except Exception:  # pragma: no cover - defensive
            pass

    def register_completion_callback(
        self, callback: CompletionCallback
    ) -> None:
        if callback in self._completion_callbacks:
            return
        self._completion_callbacks.append(callback)

    def unregister_completion_callback(
        self, callback: CompletionCallback
    ) -> None:
        try:
            self._completion_callbacks.remove(callback)
        except ValueError:
            pass

    def _notify_completion(self, invocation: GenAI) -> None:
        if not self._completion_callbacks:
            return
        callbacks = list(self._completion_callbacks)
        for callback in callbacks:
            try:
                callback.on_completion(invocation)
            except Exception:  # pragma: no cover - defensive
                continue

    def _initialize_default_callbacks(self) -> None:
        disable_defaults = is_truthy_env(
            os.getenv(
                OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS
            )
        )
        if disable_defaults:
            _LOGGER.debug(
                "Default completion callbacks disabled via %s",
                OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS,
            )
            return

        selected = parse_callback_filter(
            os.getenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS)
        )
        callbacks, seen = load_completion_callbacks(selected)
        if selected:
            missing = selected - seen
            for name in missing:
                _LOGGER.debug(
                    "Completion callback '%s' not found in entry points",
                    name,
                )
        if not callbacks:
            return

        for name, callback in callbacks:
            bound_ok = True
            binder = getattr(callback, "bind_handler", None)
            if callable(binder):
                try:
                    bound_ok = bool(binder(self))
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.warning(
                        "Completion callback '%s' failed to bind: %s",
                        name,
                        exc,
                    )
                    shutdown = getattr(callback, "shutdown", None)
                    if callable(shutdown):
                        try:
                            shutdown()
                        except Exception:  # pragma: no cover - defensive
                            pass
                    continue
            if not bound_ok:
                shutdown = getattr(callback, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:  # pragma: no cover - defensive
                        pass
                continue
            manager = getattr(callback, "manager", None)
            if manager is not None:
                self._evaluation_manager = manager
            self.register_completion_callback(callback)

    def stop_workflow(self, workflow: Workflow) -> Workflow:
        """Finalize a workflow successfully and end its span."""
        workflow.end_time = timeit.default_timer()

        workflow.sample_for_evaluation = self._should_sample_for_evaluation(
            workflow.trace_id
        )

        self._notify_completion(workflow)
        self._emitter.on_end(workflow)
        self._pop_current_span(workflow)
        self._maybe_force_flush()
        return workflow

    def fail_workflow(self, workflow: Workflow, error: Error) -> Workflow:
        """Fail a workflow and end its span with error status."""
        workflow.end_time = timeit.default_timer()
        self._emitter.on_error(error, workflow)
        self._notify_completion(workflow)
        self._pop_current_span(workflow)
        self._maybe_force_flush()
        return workflow

    # Agent lifecycle -----------------------------------------------------
    def start_agent(
        self, agent: AgentCreation | AgentInvocation
    ) -> AgentCreation | AgentInvocation:
        """Start an agent operation (create or invoke) and create a pending span entry."""
        agent._handler = self
        self._refresh_capture_content()
        _apply_genai_context(agent)
        self._inherit_parent_span(agent)
        self._maybe_mark_conversation_root(agent)
        self._emitter.on_start(agent)
        self._push_current_span(agent)
        # Push agent identity context
        if isinstance(agent, AgentInvocation):
            try:
                if agent.name:
                    self._agent_context_stack.append(
                        (agent.name, agent.agent_id)
                    )
            except Exception:  # pragma: no cover - defensive
                pass
        return agent

    def stop_agent(
        self, agent: AgentCreation | AgentInvocation
    ) -> AgentCreation | AgentInvocation:
        """Finalize an agent operation successfully and end its span."""
        agent.end_time = timeit.default_timer()

        agent.sample_for_evaluation = self._should_sample_for_evaluation(
            agent.trace_id
        )

        self._notify_completion(agent)
        self._emitter.on_end(agent)
        self._pop_current_span(agent)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        # Pop context if matches top
        if isinstance(agent, AgentInvocation):
            try:
                if self._agent_context_stack and agent.agent_id is not None:
                    top_name, top_id = self._agent_context_stack[-1]
                    if top_name == agent.name and top_id == agent.agent_id:
                        self._agent_context_stack.pop()
            except Exception:
                pass
        return agent

    def fail_agent(
        self, agent: AgentCreation | AgentInvocation, error: Error
    ) -> AgentCreation | AgentInvocation:
        """Fail an agent operation and end its span with error status."""
        agent.end_time = timeit.default_timer()
        self._emitter.on_error(error, agent)
        self._notify_completion(agent)
        self._pop_current_span(agent)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        # Pop context if this agent is active
        if isinstance(agent, AgentInvocation):
            try:
                if self._agent_context_stack and agent.agent_id is not None:
                    top_name, top_id = self._agent_context_stack[-1]
                    if top_name == agent.name and top_id == agent.agent_id:
                        self._agent_context_stack.pop()
            except Exception:
                pass
        return agent

    # Step lifecycle ------------------------------------------------------
    def start_step(self, step: Step) -> Step:
        """Start a step and create a pending span entry."""
        step._handler = self
        self._refresh_capture_content()
        _apply_genai_context(step)
        self._inherit_parent_span(step)
        self._emitter.on_start(step)
        self._push_current_span(step)
        return step

    def stop_step(self, step: Step) -> Step:
        """Finalize a step successfully and end its span."""
        step.end_time = timeit.default_timer()

        step.sample_for_evaluation = self._should_sample_for_evaluation(
            step.trace_id
        )

        self._notify_completion(step)
        self._emitter.on_end(step)
        self._pop_current_span(step)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return step

    def fail_step(self, step: Step, error: Error) -> Step:
        """Fail a step and end its span with error status."""
        step.end_time = timeit.default_timer()
        self._emitter.on_error(error, step)
        self._notify_completion(step)
        self._pop_current_span(step)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return step

    def evaluate_llm(
        self,
        invocation: LLMInvocation,
        evaluators: Optional[list[str]] = None,
    ) -> list[EvaluationResult]:
        """Proxy to EvaluationManager for running evaluators.

        Retained public signature for backward compatibility. The underlying
        implementation has been refactored into EvaluationManager to allow
        pluggable emission similar to emitters.
        """
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return []
        if evaluators:
            _LOGGER.warning(
                "Direct evaluator overrides are ignored; using configured evaluators"
            )
        return manager.evaluate_now(invocation)  # type: ignore[attr-defined]

    def evaluate_agent(
        self,
        agent: AgentInvocation,
        evaluators: Optional[list[str]] = None,
    ) -> list[EvaluationResult]:
        """Run evaluators against an AgentInvocation.

        Mirrors evaluate_llm to allow explicit agent evaluation triggering.
        """
        if not isinstance(agent, AgentInvocation):
            _LOGGER.debug(
                "Skipping agent evaluation for non-invocation type: %s",
                type(agent).__name__,
            )
            return []
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return []
        if evaluators:
            _LOGGER.warning(
                "Direct evaluator overrides are ignored; using configured evaluators"
            )
        return manager.evaluate_now(agent)  # type: ignore[attr-defined]

    def wait_for_evaluations(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending evaluations to complete, up to the specified timeout.

        This is primarily intended for use in test scenarios to ensure that
        all asynchronous evaluation steps have finished before assertions are made.
        """
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return
        manager.wait_for_all(timeout)  # type: ignore[attr-defined]

    # Internal dispatch for invocation.stop() / invocation.fail() ---------
    def _stop_invocation(self, obj: Any) -> None:
        """Internal: dispatch stop to the appropriate type-specific method.

        Called by ``GenAI.stop()`` on the invocation object.
        New-style ``GenAIInvocation`` instances handle their own stop()
        directly (no dispatch needed here); this is only used by the
        deprecated ``GenAI.stop()`` path on old dataclasses.
        """
        if isinstance(obj, GenAIInvocation):
            # New-style: invocation handles everything internally
            obj.stop()
            return
        if isinstance(obj, Workflow):
            self.stop_workflow(obj)
        elif isinstance(obj, (AgentCreation, AgentInvocation)):
            self.stop_agent(obj)
        elif isinstance(obj, Step):
            self.stop_step(obj)
        elif isinstance(obj, LLMInvocation):
            self.stop_llm(obj)
        elif isinstance(obj, EmbeddingInvocation):
            self.stop_embedding(obj)
        elif isinstance(obj, RetrievalInvocation):
            self.stop_retrieval(obj)
        elif isinstance(obj, ToolCall):
            self.stop_tool_call(obj)

    def _fail_invocation(self, obj: Any, error: Error) -> None:
        """Internal: dispatch fail to the appropriate type-specific method.

        Called by ``GenAI.fail()`` on the invocation object.
        """
        if isinstance(obj, GenAIInvocation):
            # New-style: invocation handles everything internally
            obj.fail(error)
            return
        if isinstance(obj, Workflow):
            self.fail_workflow(obj, error)
        elif isinstance(obj, (AgentCreation, AgentInvocation)):
            self.fail_agent(obj, error)
        elif isinstance(obj, Step):
            self.fail_step(obj, error)
        elif isinstance(obj, LLMInvocation):
            self.fail_llm(obj, error)
        elif isinstance(obj, EmbeddingInvocation):
            self.fail_embedding(obj, error)
        elif isinstance(obj, RetrievalInvocation):
            self.fail_retrieval(obj, error)
        elif isinstance(obj, ToolCall):
            self.fail_tool_call(obj, error)

    # Generic lifecycle API ------------------------------------------------
    def start(self, obj: Any) -> Any:
        """Generic start method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.start_workflow(obj)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.start_agent(obj)
        if isinstance(obj, Step):
            return self.start_step(obj)
        if isinstance(obj, LLMInvocation):
            return self.start_llm(obj)
        if isinstance(obj, EmbeddingInvocation):
            return self.start_embedding(obj)
        if isinstance(obj, RetrievalInvocation):
            return self.start_retrieval(obj)
        if isinstance(obj, ToolCall):
            return self.start_tool_call(obj)
        return obj

    def finish(self, obj: Any) -> Any:
        """Generic finish method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.stop_workflow(obj)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.stop_agent(obj)
        if isinstance(obj, Step):
            return self.stop_step(obj)
        if isinstance(obj, LLMInvocation):
            return self.stop_llm(obj)
        if isinstance(obj, EmbeddingInvocation):
            return self.stop_embedding(obj)
        if isinstance(obj, RetrievalInvocation):
            return self.stop_retrieval(obj)
        if isinstance(obj, ToolCall):
            return self.stop_tool_call(obj)
        return obj

    def fail(self, obj: Any, error: Error) -> Any:
        """Generic fail method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.fail_workflow(obj, error)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.fail_agent(obj, error)
        if isinstance(obj, Step):
            return self.fail_step(obj, error)
        if isinstance(obj, LLMInvocation):
            return self.fail_llm(obj, error)
        if isinstance(obj, EmbeddingInvocation):
            return self.fail_embedding(obj, error)
        if isinstance(obj, RetrievalInvocation):
            return self.fail_retrieval(obj, error)
        if isinstance(obj, ToolCall):
            return self.fail_tool_call(obj, error)
        return obj


def get_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    logger_provider: LoggerProvider | None = None,
) -> TelemetryHandler:
    """
    Returns the process-wide singleton ``TelemetryHandler`` instance.

    This is the **preferred** public API for obtaining a handler.  Both this
    function and ``TelemetryHandler(...)`` return the same singleton;
    ``TelemetryHandler.__new__`` handles the thread-safe singleton logic.
    """
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
