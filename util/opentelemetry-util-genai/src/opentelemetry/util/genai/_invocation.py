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

"""Base class for new-style GenAI invocation types.

New invocations are self-contained: they receive all required components
(emitter, agent context stack, callbacks, etc.) at construction time and
handle their own lifecycle via ``stop()`` and ``fail()`` methods.

Use the factory methods on ``TelemetryHandler`` (``start_inference``,
``start_embedding``, ``start_tool``, ``start_workflow``) rather than
constructing invocations directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import timeit
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple

from typing_extensions import Self

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import Span, SpanContext

from opentelemetry.util.genai.types import Error, ErrorClassification

_LOGGER = logging.getLogger(__name__)

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _is_async_context() -> bool:
    """Return True when called inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class GenAIInvocation:
    """Base class for all new-style GenAI invocation types.

    Manages the lifecycle of a single GenAI operation (LLM call, embedding,
    tool execution, workflow, etc.).  Invocations are self-contained: they
    hold references to the emitter pipeline, agent context stack, completion
    callbacks, and other components needed to start, stop, and fail
    telemetry spans.

    Subclasses must call ``self._start()`` at the end of ``__init__``.
    """

    def __init__(
        self,
        *,
        emitter: Any,
        agent_context_stack: List[Tuple[str, Optional[str]]],
        completion_callbacks: list,
        sampler_fn: Callable[[Optional[int]], bool],
        meter_provider: Optional[MeterProvider] = None,
        capture_refresh_fn: Optional[Callable[[], None]] = None,
        # Common fields
        provider: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        # Components (from handler)
        self._emitter = emitter
        self._agent_context_stack = agent_context_stack
        self._completion_callbacks = completion_callbacks
        self._sampler_fn = sampler_fn
        self._meter_provider = meter_provider
        self._capture_refresh_fn = capture_refresh_fn

        # Fields matching GenAI dataclass (for emitter compatibility)
        self.span: Optional[Span] = None
        self.span_context: Optional[SpanContext] = None
        self.trace_id: Optional[int] = None
        self.span_id: Optional[int] = None
        self.trace_flags: Optional[int] = None
        self.start_time: float = timeit.default_timer()
        self.end_time: Optional[float] = None
        self.parent_span: Optional[Span] = None
        self._otel_context_token: Any = None
        self.provider = provider
        self.framework = framework
        self.system = system
        self.agent_name: Optional[str] = None
        self.agent_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.conversation_root: Optional[bool] = None
        self.data_source_id: Optional[str] = None
        self.association_properties: dict[str, Any] = {}
        self.sample_for_evaluation: Optional[bool] = True
        self.evaluation_error: Optional[str] = None
        self.attributes: dict[str, Any] = attributes if attributes is not None else {}
        self.context_token: Any = None

    # -- Start helpers -------------------------------------------------------

    def _start(self) -> None:
        """Start the invocation: apply context, create span via emitter.

        Called by subclass ``__init__`` after all type-specific fields are set.
        """
        # Refresh content capture settings
        if self._capture_refresh_fn is not None:
            try:
                self._capture_refresh_fn()
            except Exception:  # pragma: no cover
                pass

        # Apply GenAI context from contextvars
        self._apply_genai_context()

        # Inherit agent context from stack
        self._apply_agent_context()

        # Inherit parent span if not set
        self._inherit_parent_span()

        # Emit start (creates span)
        self._emitter.on_start(self)

        # Push current span for child resolution
        self._push_current_span()

    def _apply_genai_context(self) -> None:
        """Apply GenAI context (conversation_id, association_properties) from contextvars."""
        from opentelemetry.util.genai.handler import (
            _apply_genai_context as _apply_ctx,
        )

        _apply_ctx(self)

    def _apply_agent_context(self) -> None:
        """Inherit agent_name and agent_id from agent context stack."""
        if (
            not self.agent_name or not self.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not self.agent_name:
                self.agent_name = top_name
            if not self.agent_id:
                self.agent_id = top_id

    def _inherit_parent_span(self) -> None:
        """If parent_span is unset, inherit from the current GenAI span."""
        from opentelemetry.util.genai.handler import _current_genai_span

        if self.parent_span is None:
            current = _current_genai_span.get()
            if current is not None:
                self.parent_span = current

    def _push_current_span(self) -> None:
        """Track this span as current for child resolution."""
        from opentelemetry.util.genai.handler import _current_genai_span

        span = self.span
        if span is not None:
            _current_genai_span.set(span)
            if not _is_async_context():
                ctx = trace.set_span_in_context(span)
                self._otel_context_token = context_api.attach(ctx)

    def _pop_current_span(self) -> None:
        """Restore current span to parent (effectively a pop)."""
        from opentelemetry.util.genai.handler import _current_genai_span

        _current_genai_span.set(self.parent_span)
        token = self._otel_context_token
        if token is not None:
            try:
                context_api._RUNTIME_CONTEXT.detach(token)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    # -- Finish helpers ------------------------------------------------------

    def _finish(self, error: Optional[Error] = None) -> None:
        """Finalize the invocation and end its span.

        Sets end_time, evaluation sampling, emits telemetry, fires callbacks,
        restores span context, and flushes metrics.
        """
        self.end_time = timeit.default_timer()

        self.sample_for_evaluation = self._sampler_fn(self.trace_id)

        if error is not None:
            self._emitter.on_error(error, self)
        else:
            self._emitter.on_end(self)

        self._notify_completion()
        self._pop_current_span()
        self._maybe_force_flush()

    def _notify_completion(self) -> None:
        """Fire completion callbacks."""
        if not self._completion_callbacks:
            return
        callbacks = list(self._completion_callbacks)
        for callback in callbacks:
            try:
                callback.on_completion(self)
            except Exception:  # pragma: no cover
                continue

    def _maybe_force_flush(self) -> None:
        """Force flush metrics if a meter provider is available."""
        if self._meter_provider is not None:
            try:
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass

    # -- Public API ----------------------------------------------------------

    def stop(self) -> None:
        """Finalize the invocation successfully and end its span."""
        self._finish()

    def fail(self, error: Error | BaseException) -> None:
        """Fail the invocation and end its span with error status.

        Accepts either an ``Error`` dataclass or a raw exception (which
        is wrapped in ``Error`` automatically).
        """
        if isinstance(error, BaseException):
            error = Error(
                message=str(error),
                type=type(error),
                classification=ErrorClassification.REAL_ERROR,
            )
        self._finish(error)

    @contextmanager
    def _managed(self) -> Iterator[Self]:
        """Context manager that calls ``stop()`` on success or ``fail()`` on exception."""
        try:
            yield self  # type: ignore[misc]
        except Exception as exc:
            self.fail(exc)
            raise
        self.stop()

    # -- Semconv helpers (for emitter compatibility) -------------------------

    def semantic_convention_attributes(self) -> dict[str, Any]:
        """Return semantic convention attributes for this invocation.

        Subclasses should override to include type-specific attributes.
        """
        from opentelemetry.util.genai.attributes import (
            GEN_AI_ASSOCIATION_PROPERTIES_PREFIX,
        )
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )

        result: dict[str, Any] = {}
        # Common attributes
        _optional = (
            (GenAIAttributes.GEN_AI_PROVIDER_NAME, self.provider),
            (GenAIAttributes.GEN_AI_AGENT_NAME, self.agent_name),
            (GenAIAttributes.GEN_AI_AGENT_ID, self.agent_id),
            (GenAIAttributes.GEN_AI_SYSTEM, self.system),
            (GenAIAttributes.GEN_AI_CONVERSATION_ID, self.conversation_id),
            (GenAIAttributes.GEN_AI_DATA_SOURCE_ID, self.data_source_id),
        )
        for key, value in _optional:
            if value is not None:
                result[key] = value
        if self.conversation_root is not None:
            result["gen_ai.conversation_root"] = self.conversation_root

        # Association properties
        for key, value in self.association_properties.items():
            result[f"{GEN_AI_ASSOCIATION_PROPERTIES_PREFIX}.{key}"] = value

        return result
