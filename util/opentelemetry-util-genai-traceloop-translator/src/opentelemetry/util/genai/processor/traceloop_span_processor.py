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

import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)
from opentelemetry.util.genai.span_context import extract_span_context
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    Workflow,
)

from .content_normalizer import normalize_traceloop_content
from .message_reconstructor import reconstruct_messages_from_traceloop

_ENV_RULES = "OTEL_GENAI_SPAN_TRANSFORM_RULES"

# LLM span detection constants
_LLM_OPERATIONS = [
    "chat",
    "completion",
    "embedding",
    "embed",
    "invoke_agent",
    "create_agent",
]
_EXCLUDE_SPAN_PATTERNS = [
    "__start__",
    "__end__",
    "should_continue",
    "model_to_tools",
    "tools_to_model",
    # Exclude Deepeval evaluation spans (prevent recursive evaluation)
    "Run evaluate",
    "Ran evaluate",
    "Ran test case",
    "Bias",
    "Toxicity",
    "Relevance",
    "Hallucination",
    "Sentiment",
    "deepeval",
]
_LLM_API_CALL_PATTERNS = [
    ".chat",  # ChatOpenAI.chat, ChatAnthropic.chat, etc.
    "openai.chat",
    "anthropic.chat",
    ".completion",
    "completions",
]
_LLM_MODEL_ATTRIBUTES = [
    "gen_ai.request.model",
    "llm.request.model",
]


@dataclass
class TransformationRule:
    """Represents a single conditional transformation rule.

    Fields map closely to the JSON structure accepted via the environment
    variable. All fields are optional; empty rule never matches.
    """

    match_name: Optional[str] = None  # glob pattern (e.g. "chat *")
    match_scope: Optional[str] = None  # regex or substring (case-insensitive)
    match_attributes: Dict[str, Optional[str]] = field(default_factory=dict)

    attribute_transformations: Dict[str, Any] = field(default_factory=dict)
    name_transformations: Dict[str, str] = field(default_factory=dict)
    traceloop_attributes: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self, span: ReadableSpan
    ) -> bool:  # pragma: no cover - simple logic
        if self.match_name:
            if not fnmatch.fnmatch(span.name, self.match_name):
                return False
        if self.match_scope:
            scope = getattr(span, "instrumentation_scope", None)
            scope_name = getattr(scope, "name", "") if scope else ""
            pattern = self.match_scope
            # Accept either regex (contains meta chars) or simple substring
            try:
                if any(ch in pattern for ch in ".^$|()[]+?\\"):
                    if not re.search(pattern, scope_name, re.IGNORECASE):
                        return False
                else:
                    if pattern.lower() not in scope_name.lower():
                        return False
            except re.error:
                # Bad regex – treat as non-match but log once
                logging.warning(
                    "[TL_PROCESSOR] Invalid regex in match_scope: %s", pattern
                )
                return False
        if self.match_attributes:
            for k, expected in self.match_attributes.items():
                if k not in span.attributes:
                    return False
                if expected is not None and str(span.attributes.get(k)) != str(
                    expected
                ):
                    return False
        return True


def _load_rules_from_env() -> List[TransformationRule]:
    raw = os.getenv(_ENV_RULES)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        rules_spec = data.get("rules") if isinstance(data, dict) else None
        if not isinstance(rules_spec, list):
            logging.warning(
                "[TL_PROCESSOR] %s must contain a 'rules' list", _ENV_RULES
            )
            return []
        rules: List[TransformationRule] = []
        for r in rules_spec:
            if not isinstance(r, dict):
                continue
            match = (
                r.get("match", {}) if isinstance(r.get("match"), dict) else {}
            )
            rules.append(
                TransformationRule(
                    match_name=match.get("name"),
                    match_scope=match.get("scope"),
                    match_attributes=match.get("attributes", {}) or {},
                    attribute_transformations=r.get(
                        "attribute_transformations", {}
                    )
                    or {},
                    name_transformations=r.get("name_transformations", {})
                    or {},
                    traceloop_attributes=r.get("traceloop_attributes", {})
                    or {},
                )
            )
        return rules
    except Exception as exc:  # broad: we never want to break app startup
        logging.warning(
            "[TL_PROCESSOR] Failed to parse %s: %s", _ENV_RULES, exc
        )
        return []


class TraceloopSpanProcessor(SpanProcessor):
    """
    A span processor that automatically applies transformation rules to spans.

    This processor can be added to your TracerProvider to automatically transform
    all spans according to your transformation rules.
    """

    def __init__(
        self,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
        span_filter: Optional[Callable[[ReadableSpan], bool]] = None,
        rules: Optional[List[TransformationRule]] = None,
        load_env_rules: bool = True,
        telemetry_handler: Optional[TelemetryHandler] = None,
        mutate_original_span: bool = True,
    ):
        """
        Initialize the Traceloop span processor.

        Args:
            attribute_transformations: Rules for transforming span attributes
            name_transformations: Rules for transforming span names
            traceloop_attributes: Additional Traceloop-specific attributes to add
            span_filter: Optional filter function to determine which spans to transform
            rules: Optional list of TransformationRule objects for conditional transformations
            load_env_rules: Whether to load transformation rules from OTEL_GENAI_SPAN_TRANSFORM_RULES
            telemetry_handler: Optional TelemetryHandler for emitting transformed spans
            mutate_original_span: Whether to mutate original spans at the processor level.
                This flag works in conjunction with the mutate_original_span field on
                individual GenAI objects. Both must be True for mutation to occur.
                Default is True for backward compatibility.
        """
        self.attribute_transformations = attribute_transformations or {}
        self.name_transformations = name_transformations or {}
        self.traceloop_attributes = traceloop_attributes or {}
        self.span_filter = span_filter or self._default_span_filter
        # Load rule set (env + explicit). Explicit rules first for precedence.
        env_rules = _load_rules_from_env() if load_env_rules else []
        self.rules: List[TransformationRule] = list(rules or []) + env_rules
        self.telemetry_handler = telemetry_handler
        self.mutate_original_span = mutate_original_span
        if self.rules:
            logging.getLogger(__name__).debug(
                "TraceloopSpanProcessor loaded %d transformation rules (explicit=%d env=%d)",
                len(self.rules),
                len(rules or []),
                len(env_rules),
            )
        self._processed_span_ids = set()
        # Track synthetic span IDs to prevent recursion (since ReadableSpan attributes are immutable snapshots)
        self._synthetic_span_ids: set[int] = set()
        # Mapping from original span_id to translated INVOCATION (not span) for parent-child relationship preservation
        self._original_to_translated_invocation: Dict[int, Any] = {}
        # Buffer spans to process them in the correct order (parents before children)
        self._span_buffer: List[ReadableSpan] = []
        self._processing_buffer = False
        # Cache reconstructed messages to avoid double reconstruction
        self._message_cache: Dict[int, tuple] = {}

    def _default_span_filter(self, span: ReadableSpan) -> bool:
        """Default filter: Transform spans that look like LLM/AI calls.

        Filters out spans that don't appear to be LLM-related while keeping
        Traceloop task/workflow spans for transformation.
        """
        if not span.name:
            return False

        # Always process Traceloop task/workflow spans (they need transformation)
        if span.attributes:
            span_kind = span.attributes.get("traceloop.span.kind")
            if span_kind in ("task", "workflow", "tool", "agent"):
                return True

        # Check for common LLM/AI span indicators
        llm_indicators = [
            "chat",
            "completion",
            "llm",
            # "ai",
            "gpt",
            "claude",
            "gemini",
            "openai",
            "anthropic",
            "cohere",
            "huggingface",
        ]

        span_name_lower = span.name.lower()
        for indicator in llm_indicators:
            if indicator in span_name_lower:
                return True

        # Check attributes for AI/LLM markers (if any attributes present)
        if span.attributes:
            # Check for traceloop entity attributes
            if (
                "traceloop.entity.input" in span.attributes
                or "traceloop.entity.output" in span.attributes
            ):
                # We already filtered task/workflow spans above, so if we get here
                # it means it has model data
                return True
            # Check for other AI/LLM markers
            for attr_key in span.attributes.keys():
                attr_key_lower = str(attr_key).lower()
                if any(
                    marker in attr_key_lower
                    for marker in ["llm", "ai", "gen_ai", "model"]
                ):
                    return True
        return False

    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        """Called when a span is started."""
        pass

    def _process_span_translation(self, span: ReadableSpan) -> Optional[Any]:
        """Process a single span translation with proper parent mapping.

        Returns the invocation object if a translation was created, None otherwise.
        """
        logger = logging.getLogger(__name__)

        # Skip synthetic spans we already produced (recursion guard) - use different sentinel
        # NOTE: _traceloop_processed is set by mutation, _traceloop_translated is set by translation
        if span.attributes and "_traceloop_translated" in span.attributes:
            return None

        # Check if this span should be transformed
        if not self.span_filter(span):
            logger.debug("[TL_PROCESSOR] Span filtered: name=%s", span.name)
            return None

        logger.debug(
            "[TL_PROCESSOR] Translating span: name=%s, kind=%s",
            span.name,
            span.attributes.get("traceloop.span.kind")
            if span.attributes
            else None,
        )

        # avoid emitting multiple synthetic spans if on_end invoked repeatedly.
        span_id_int = getattr(getattr(span, "context", None), "span_id", None)
        if span_id_int is not None:
            if span_id_int in self._processed_span_ids:
                return None
            self._processed_span_ids.add(span_id_int)

        # Determine which transformation set to use
        applied_rule: Optional[TransformationRule] = None
        for rule in self.rules:
            try:
                if rule.matches(span):
                    applied_rule = rule
                    break
            except Exception as match_err:  # pragma: no cover - defensive
                logging.warning(
                    "[TL_PROCESSOR] Rule match error: %s", match_err
                )

        sentinel = {"_traceloop_processed": True}
        # Decide which transformation config to apply
        if applied_rule is not None:
            attr_tx = applied_rule.attribute_transformations
            name_tx = applied_rule.name_transformations
            extra_tl_attrs = {
                **applied_rule.traceloop_attributes,
                **sentinel,
            }
        else:
            attr_tx = self.attribute_transformations
            name_tx = self.name_transformations
            extra_tl_attrs = {**self.traceloop_attributes, **sentinel}

        # Build invocation (mutation already happened in on_end before this method)
        invocation = self._build_invocation(
            span,
            attribute_transformations=attr_tx,
            name_transformations=name_tx,
            traceloop_attributes=extra_tl_attrs,
        )

        # If invocation is None, it means we couldn't get messages - skip this span
        if invocation is None:
            logger.debug(
                "[TL_PROCESSOR] Skipping span translation - invocation creation returned None: %s",
                span.name,
            )
            return None

        invocation.attributes.setdefault("_traceloop_processed", True)

        # Always emit via TelemetryHandler
        handler = self.telemetry_handler or get_telemetry_handler()
        try:
            # Find the translated parent span if the original span has a parent
            parent_context = None
            if span.parent:
                parent_span_id = getattr(span.parent, "span_id", None)
                if (
                    parent_span_id
                    and parent_span_id
                    in self._original_to_translated_invocation
                ):
                    # We found the translated invocation of the parent - use its span
                    translated_parent_invocation = (
                        self._original_to_translated_invocation[parent_span_id]
                    )
                    translated_parent_span = getattr(
                        translated_parent_invocation, "span", None
                    )
                    if (
                        translated_parent_span
                        and hasattr(translated_parent_span, "is_recording")
                        and translated_parent_span.is_recording()
                    ):
                        parent_context = set_span_in_context(
                            translated_parent_span
                        )

            original_span_id = getattr(
                getattr(span, "context", None), "span_id", None
            )

            invocation.parent_context = parent_context
            handler.start(invocation)

            # CRITICAL: Track synthetic span ID IMMEDIATELY after creation to prevent recursion
            # We use a set instead of span attributes because ReadableSpan is immutable
            synthetic_span = getattr(invocation, "span", None)
            if synthetic_span:
                # Try to get span ID from context
                synthetic_span_id = None
                try:
                    if hasattr(synthetic_span, "get_span_context"):
                        span_ctx = synthetic_span.get_span_context()
                        synthetic_span_id = (
                            span_ctx.span_id if span_ctx else None
                        )
                except Exception:
                    pass

                if not synthetic_span_id:
                    # Try alternative way to get span ID
                    try:
                        span_ctx = extract_span_context(synthetic_span)
                        synthetic_span_id = (
                            span_ctx.span_id if span_ctx else None
                        )
                    except Exception:
                        pass

                if synthetic_span_id:
                    self._synthetic_span_ids.add(synthetic_span_id)
                    logger.debug(
                        "[TL_PROCESSOR] Marked synthetic span ID=%s for skipping",
                        synthetic_span_id,
                    )

                # Also set attribute as defense-in-depth
                if (
                    hasattr(synthetic_span, "set_attribute")
                    and synthetic_span.is_recording()
                ):
                    try:
                        synthetic_span.set_attribute(
                            "_traceloop_translated", True
                        )
                    except Exception:
                        pass

            # Store the mapping from original span_id to translated INVOCATION (we'll close it later)
            if original_span_id:
                self._original_to_translated_invocation[original_span_id] = (
                    invocation
                )
            # DON'T call stop_llm yet - we'll do that after processing all children
            return invocation
        except Exception as emit_err:  # pragma: no cover - defensive
            logging.getLogger(__name__).warning(
                "Telemetry handler emission failed: %s", emit_err
            )
            return None

    def _should_skip_span(
        self, span: ReadableSpan, span_id: Optional[int] = None
    ) -> bool:
        """
        Check if a span should be skipped from processing.

        Returns True if the span should be skipped, False otherwise.
        """
        _logger = logging.getLogger(__name__)

        if not span or not span.name:
            return True

        # Skip synthetic spans we created (check span ID in set)
        if span_id and span_id in self._synthetic_span_ids:
            _logger.debug(
                "[TL_PROCESSOR] Skipping synthetic span (ID in set): %s",
                span.name,
            )
            return True

        # Fallback: Also check attributes for defense-in-depth
        if span.attributes and "_traceloop_translated" in span.attributes:
            _logger.debug(
                "[TL_PROCESSOR] Skipping synthetic span (attribute): %s",
                span.name,
            )
            return True

        # Skip already processed spans
        if span.attributes and "_traceloop_processed" in span.attributes:
            _logger.debug(
                "[TL_PROCESSOR] Skipping already processed span: %s", span.name
            )
            return True

        return False

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span is ended. Mutate immediately, then process based on span type.

        HYBRID APPROACH:
        1. ALL spans get attribute translation immediately (via _mutate_span_if_needed)
        2. LLM spans get processed immediately for evaluations
        3. Non-LLM spans are buffered for optional batch processing
        """
        _logger = logging.getLogger(__name__)

        try:
            # STEP 0: Check if we should skip this span (synthetic, already processed, etc.)
            span_id = getattr(getattr(span, "context", None), "span_id", None)
            if self._should_skip_span(span, span_id):
                return

            # STEP 1: Always mutate immediately (ALL spans get attribute translation)
            self._mutate_span_if_needed(span)

            # STEP 1.5: Skip evaluation-related spans entirely (don't buffer AND don't export)
            # These are Deepeval's internal spans that should never be processed or exported
            span_name = span.name or ""
            for exclude_pattern in _EXCLUDE_SPAN_PATTERNS:
                if exclude_pattern.lower() in span_name.lower():
                    _logger.debug(
                        "[TL_PROCESSOR] Span excluded (will not export): pattern='%s', span=%s",
                        exclude_pattern,
                        span_name,
                    )
                    # CRITICAL: Mark span as non-sampled to prevent export
                    # This prevents the span from being sent to the backend
                    if hasattr(span, "_context") and hasattr(
                        span._context, "_trace_flags"
                    ):  # type: ignore
                        try:
                            # Set trace flags to 0 (not sampled)
                            span._context._trace_flags = 0  # type: ignore
                            _logger.debug(
                                "[TL_PROCESSOR] Marked span as non-sampled: %s",
                                span_name,
                            )
                        except Exception as e:
                            _logger.debug(
                                "[TL_PROCESSOR] Could not mark span as non-sampled: %s",
                                e,
                            )
                    return

            # STEP 2: Check if this is an LLM span that needs evaluation
            if self._is_llm_span(span):
                _logger.debug(
                    "[TL_PROCESSOR] LLM span detected: %s, processing for evaluations",
                    span.name,
                )
                # Build invocation from mutated span data (no synthetic span creation)
                # The mutation already happened in step 1, so we just build the invocation
                # and call handler.finish() directly to trigger evaluations
                invocation = self._build_invocation(
                    span,
                    attribute_transformations=self.attribute_transformations,
                    name_transformations=self.name_transformations,
                    traceloop_attributes=self.traceloop_attributes,
                )
                if invocation:
                    # Attach the original (mutated) span to the invocation
                    # This is normally done by handler.start(), but we're skipping that
                    # to avoid creating a synthetic span
                    invocation.span = span  # type: ignore[attr-defined]

                    # Extract trace context from the original span
                    span_context = getattr(span, "context", None)
                    trace_id = getattr(span_context, "trace_id", None)
                    span_id_val = getattr(span_context, "span_id", None)

                    # Set trace_id on invocation (needed for sampling)
                    invocation.trace_id = trace_id
                    invocation.span_id = span_id_val

                    # Set timing info (use span's timing if available)
                    if hasattr(span, "_start_time") and span._start_time:  # type: ignore[attr-defined]
                        invocation.start_time = (
                            span._start_time / 1e9
                        )  # Convert ns to seconds  # type: ignore[attr-defined]

                    # DEBUG: Verify messages are present before calling finish
                    input_count = (
                        len(invocation.input_messages)
                        if hasattr(invocation, "input_messages")
                        and invocation.input_messages
                        else 0
                    )
                    output_count = (
                        len(invocation.output_messages)
                        if hasattr(invocation, "output_messages")
                        and invocation.output_messages
                        else 0
                    )
                    _logger.debug(
                        "[TL_PROCESSOR] Calling finish with messages: input=%d, output=%d, span=%s",
                        input_count,
                        output_count,
                        span.name,
                    )
                    if input_count == 0 and output_count == 0:
                        _logger.warning(
                            "[TL_PROCESSOR] WARNING: No messages on invocation before finish! span=%s",
                            span.name,
                        )

                    # Close the invocation to trigger core lifecycle handling
                    # This will call the appropriate stop_* method and emit spans/metrics.
                    handler = self.telemetry_handler or get_telemetry_handler()
                    try:
                        handler.finish(invocation)
                        _logger.debug(
                            "[TL_PROCESSOR] LLM/Agent invocation completed: %s, sampled=%s",
                            span.name,
                            getattr(invocation, "sample_for_evaluation", None),
                        )

                        # If this invocation represents an agent call (invoke_agent),
                        # explicitly trigger agent-level evaluations so that
                        # gen_ai.evaluation.result events can be attached to the
                        # agent span itself, in addition to any LLM-level evaluations.
                        if isinstance(invocation, AgentInvocation):  # type: ignore[attr-defined]
                            try:
                                handler.evaluate_agent(invocation)
                                _logger.debug(
                                    "[TL_PROCESSOR] Agent invocation evaluated: %s",
                                    span.name,
                                )
                            except (
                                Exception
                            ) as eval_err:  # pragma: no cover - defensive
                                _logger.warning(
                                    "[TL_PROCESSOR] Failed to evaluate AgentInvocation: %s",
                                    eval_err,
                                )

                    except Exception as stop_err:
                        _logger.warning(
                            "[TL_PROCESSOR] Failed to finish invocation: %s",
                            stop_err,
                        )
                else:
                    _logger.info(
                        "[TL_PROCESSOR] Skipped LLM span (no invocation created - missing messages): %s",
                        span.name,
                    )
                    return  # Exit early, don't try to process further
            else:
                # Non-LLM spans (tasks, workflows, tools) - buffer for optional batch processing
                _logger.debug(
                    "[TL_PROCESSOR] Non-LLM span buffered: %s (buffer_size=%d)",
                    span.name,
                    len(self._span_buffer) + 1,
                )
                self._span_buffer.append(span)

                # Process buffer when root span arrives (optional, for synthetic spans of workflows)
                if span.parent is None and not self._processing_buffer:
                    _logger.debug(
                        "[TL_PROCESSOR] Root span detected, processing buffered spans (count=%d)",
                        len(self._span_buffer),
                    )
                    self._processing_buffer = True
                    try:
                        spans_to_process = self._sort_spans_by_hierarchy(
                            self._span_buffer
                        )

                        for buffered_span in spans_to_process:
                            # Skip spans that should not be processed
                            buffered_span_id = getattr(
                                getattr(buffered_span, "context", None),
                                "span_id",
                                None,
                            )
                            if self._should_skip_span(
                                buffered_span, buffered_span_id
                            ):
                                continue

                            # Non-LLM spans (workflows, tasks, tools) don't need synthetic spans
                            # They're already mutated and will be exported as-is
                            # We only log that they were processed
                            _logger.debug(
                                "[TL_PROCESSOR] Buffered span processed (mutation only): %s",
                                buffered_span.name,
                            )

                        self._span_buffer.clear()
                        self._original_to_translated_invocation.clear()
                    finally:
                        self._processing_buffer = False

        except Exception as e:
            # Don't let transformation errors break the original span processing
            logging.warning("[TL_PROCESSOR] Span transformation failed: %s", e)

    def _sort_spans_by_hierarchy(
        self, spans: List[ReadableSpan]
    ) -> List[ReadableSpan]:
        """Sort spans so parents come before children."""
        # Build a map of span_id to span
        span_map = {}
        for s in spans:
            span_id = getattr(getattr(s, "context", None), "span_id", None)
            if span_id:
                span_map[span_id] = s

        # Build dependency graph: child -> parent
        result = []
        visited = set()

        def visit(span: ReadableSpan) -> None:
            span_id = getattr(getattr(span, "context", None), "span_id", None)
            if not span_id or span_id in visited:
                return

            # Visit parent first
            if span.parent:
                parent_id = getattr(span.parent, "span_id", None)
                if parent_id and parent_id in span_map:
                    visit(span_map[parent_id])

            # Then add this span
            visited.add(span_id)
            result.append(span)

        # Visit all spans
        for span in spans:
            visit(span)

        return result

    def shutdown(self) -> None:
        """Called when the tracer provider is shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_llm_span(self, span: ReadableSpan) -> bool:
        """
        Detect if this is an actual LLM API call span that should trigger evaluations.

        Simplified logic: Check if gen_ai.operation.name contains "chat" or other LLM operations
        (including "invoke_agent" and "create_agent").
        This is the most reliable way to identify actual LLM API calls vs orchestration spans.

        This avoids creating synthetic spans and running evaluations on workflow/task
        orchestration spans, significantly reducing span explosion.

        Returns True ONLY for actual LLM/Agent call spans.
        Returns False for workflow orchestration, utility tasks, routing, etc.
        """
        _logger = logging.getLogger(__name__)

        if not span or not span.attributes:
            return False

        # Skip synthetic spans we already created (recursion guard)
        if span.attributes and "_traceloop_translated" in span.attributes:
            return False

        # CRITICAL: Exclude evaluation-related spans (prevent recursive evaluation)
        # Deepeval creates spans like "Run evaluate()", "Bias", "Toxicity", etc.
        # These should NEVER be queued for evaluation
        span_name = span.name or ""
        for exclude_pattern in _EXCLUDE_SPAN_PATTERNS:
            if exclude_pattern.lower() in span_name.lower():
                _logger.debug(
                    "[TL_PROCESSOR] Span excluded (matches pattern '%s'): name=%s",
                    exclude_pattern,
                    span_name,
                )
                return False

        # ONLY CHECK: gen_ai.operation.name attribute (set during mutation in on_end)
        # Since _mutate_span_if_needed() is called BEFORE _is_llm_span() in on_end(),
        # ALL spans will have gen_ai.operation.name if they're LLM operations.
        # No fallback checks needed - if it doesn't have this attribute, it's not an LLM span.
        operation_name = span.attributes.get("gen_ai.operation.name")
        if operation_name:
            # Only trigger on actual LLM operations: chat, completion, embedding
            if any(
                op in str(operation_name).lower() for op in _LLM_OPERATIONS
            ):
                _logger.debug(
                    "[TL_PROCESSOR] LLM span detected (gen_ai.operation.name=%s): name=%s",
                    operation_name,
                    span.name,
                )
                return True
            else:
                # Has operation name but not an LLM operation (e.g., "workflow", "task", "tool")
                _logger.debug(
                    "[TL_PROCESSOR] Non-LLM operation (gen_ai.operation.name=%s): name=%s",
                    operation_name,
                    span.name,
                )
                return False

        # No gen_ai.operation.name means it wasn't transformed or doesn't match our rules
        _logger.debug(
            "[TL_PROCESSOR] Span skipped (no gen_ai.operation.name): name=%s",
            span.name,
        )
        return False

    def _extract_indexed_messages(
        self, attrs: dict, prefix: str, direction: str
    ) -> list:
        """
        Extract messages from indexed attributes like gen_ai.prompt.0.content, gen_ai.prompt.0.role.

        Returns list of InputMessage or OutputMessage objects.
        """
        messages = []
        idx = 0

        while True:
            role_key = f"{prefix}.{idx}.role"
            content_key = f"{prefix}.{idx}.content"

            role = attrs.get(role_key)
            content = attrs.get(content_key)

            if content is None:
                break  # No more messages at this index

            role = role or ("user" if direction == "input" else "assistant")

            if direction == "input":
                messages.append(
                    InputMessage(
                        role=role,
                        parts=[Text(content=str(content), type="text")],
                    )
                )
            else:
                # For output, also get finish_reason if available
                finish_reason = attrs.get(
                    f"{prefix}.{idx}.finish_reason", "stop"
                )
                messages.append(
                    OutputMessage(
                        role=role,
                        parts=[Text(content=str(content), type="text")],
                        finish_reason=finish_reason,
                    )
                )

            idx += 1

        return messages

    def _reconstruct_and_set_messages(
        self,
        original_attrs: dict,
        mutated_attrs: dict,
        span_name: str,
        span_id: Optional[int] = None,
    ) -> Optional[tuple]:
        """
        Reconstruct messages from Traceloop format and set them as gen_ai.* attributes.

        This ensures ALL spans have gen_ai.input.messages and gen_ai.output.messages
        in OTel format, not just spans processed for evaluation.

        Returns the reconstructed messages (input_messages, output_messages) for caching.
        """
        _logger = logging.getLogger(__name__)

        # Extract message data from various sources
        # Try multiple attribute names from different instrumentation sources:
        # 1. gen_ai.* (OTel GenAI format - already transformed or from OTel instrumentation)
        # 2. traceloop.entity.* (Traceloop SDK for LangChain/workflows)
        # 3. gen_ai.prompt/completion (OpenLLMetry OpenAI instrumentation)
        # 4. llm.prompts/completions (older OpenAI instrumentation format)
        # 5. gen_ai.content.* (another format variant)
        original_input_data = (
            mutated_attrs.get("gen_ai.input.messages")
            or original_attrs.get("gen_ai.input.messages")
            or original_attrs.get("gen_ai.input.message")
            or original_attrs.get("traceloop.entity.input")
            or original_attrs.get("gen_ai.prompt")
            or original_attrs.get("llm.prompts")
            or original_attrs.get("gen_ai.content.prompt")
        )
        original_output_data = (
            mutated_attrs.get("gen_ai.output.messages")
            or original_attrs.get("gen_ai.output.messages")
            or original_attrs.get("gen_ai.output.message")
            or original_attrs.get("traceloop.entity.output")
            or original_attrs.get("gen_ai.completion")
            or original_attrs.get("llm.completions")
            or original_attrs.get("gen_ai.content.completion")
        )

        # Check for indexed format (OpenLLMetry/Traceloop OpenAI format):
        # gen_ai.prompt.0.role, gen_ai.prompt.0.content, gen_ai.completion.0.role, etc.
        has_indexed_prompt = "gen_ai.prompt.0.content" in original_attrs
        has_indexed_completion = (
            "gen_ai.completion.0.content" in original_attrs
        )

        # Debug: log what we found
        _logger.debug(
            "[TL_PROCESSOR] _reconstruct_and_set_messages: span=%s, span_id=%s, "
            "has_input=%s, has_output=%s, has_indexed_prompt=%s, has_indexed_completion=%s",
            span_name,
            span_id,
            original_input_data is not None,
            original_output_data is not None,
            has_indexed_prompt,
            has_indexed_completion,
        )

        # If no scalar data but we have indexed format, we can still process
        if (
            not original_input_data
            and not original_output_data
            and not has_indexed_prompt
            and not has_indexed_completion
        ):
            _logger.debug(
                "[TL_PROCESSOR] No message data found in span attrs for reconstruction, "
                "available keys: %s",
                list(original_attrs.keys())[:15],
            )
            return None  # Nothing to reconstruct

        try:
            input_messages = None
            output_messages = None

            # FIRST: Try indexed format (OpenLLMetry/Traceloop OpenAI format)
            # This is the most common format for OpenAI spans from Traceloop
            if has_indexed_prompt:
                input_messages = self._extract_indexed_messages(
                    original_attrs, "gen_ai.prompt", "input"
                )
                _logger.debug(
                    "[TL_PROCESSOR] Extracted %d input messages from indexed format",
                    len(input_messages) if input_messages else 0,
                )

            if has_indexed_completion:
                output_messages = self._extract_indexed_messages(
                    original_attrs, "gen_ai.completion", "output"
                )
                _logger.debug(
                    "[TL_PROCESSOR] Extracted %d output messages from indexed format",
                    len(output_messages) if output_messages else 0,
                )

            # SECOND: If no indexed messages, try reconstructing from scalar data
            if (
                not input_messages
                and not output_messages
                and (original_input_data or original_output_data)
            ):
                # Try to reconstruct LangChain messages from Traceloop JSON format
                lc_input, lc_output = reconstruct_messages_from_traceloop(
                    original_input_data, original_output_data
                )

                # Convert to GenAI SDK format (with .parts containing Text objects)
                # This is the format DeepEval expects: InputMessage/OutputMessage with Text objects
                input_messages = self._convert_langchain_to_genai_messages(
                    lc_input, "input"
                )
                output_messages = self._convert_langchain_to_genai_messages(
                    lc_output, "output"
                )

            if not input_messages and original_input_data:
                if isinstance(original_input_data, str):
                    # Check if it's a JSON array (already formatted)
                    try:
                        parsed = json.loads(original_input_data)
                        if isinstance(parsed, list) and parsed:
                            # Already a JSON array - convert to InputMessage objects
                            input_messages = []
                            for msg in parsed:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "user")
                                    parts = msg.get("parts", [])
                                    if parts and isinstance(parts, list):
                                        content = (
                                            parts[0].get("content", "")
                                            if isinstance(parts[0], dict)
                                            else str(parts[0])
                                        )
                                    else:
                                        content = msg.get("content", str(msg))
                                    input_messages.append(
                                        InputMessage(
                                            role=role,
                                            parts=[
                                                Text(
                                                    content=content,
                                                    type="text",
                                                )
                                            ],
                                        )
                                    )
                    except json.JSONDecodeError:
                        pass

                if not input_messages:
                    # Traceloop stores raw input string in kwargs
                    try:
                        parsed = json.loads(original_input_data)
                        if isinstance(parsed, dict) and "kwargs" in parsed:
                            content = parsed["kwargs"]
                            # Convert dict/kwargs to string representation for Agent input
                            if isinstance(content, (dict, list)):
                                content_str = json.dumps(content)
                            else:
                                content_str = str(content)

                            input_messages = [
                                InputMessage(
                                    role="user",
                                    parts=[
                                        Text(
                                            content=content_str,
                                            type="text",
                                        )
                                    ],
                                )
                            ]
                        elif isinstance(parsed, dict) and "args" in parsed:
                            # Handle args list (positional arguments)
                            args = parsed["args"]
                            if args and isinstance(args, list):
                                content_parts = []
                                for arg in args:
                                    if isinstance(arg, (dict, list)):
                                        content_parts.append(json.dumps(arg))
                                    else:
                                        content_parts.append(str(arg))
                                content_str = " ".join(content_parts)

                                input_messages = [
                                    InputMessage(
                                        role="user",
                                        parts=[
                                            Text(
                                                content=content_str,
                                                type="text",
                                            )
                                        ],
                                    )
                                ]
                    except json.JSONDecodeError:
                        # Plain text string - create single InputMessage
                        input_messages = [
                            InputMessage(
                                role="user",
                                parts=[
                                    Text(
                                        content=original_input_data,
                                        type="text",
                                    )
                                ],
                            )
                        ]
                        _logger.debug(
                            "[TL_PROCESSOR] Created InputMessage from plain string: %s...",
                            original_input_data[:50],
                        )

            if not output_messages and original_output_data:
                if isinstance(original_output_data, str):
                    # Check if it's a JSON array (already formatted)
                    try:
                        parsed = json.loads(original_output_data)
                        if isinstance(parsed, list) and parsed:
                            # Already a JSON array - convert to OutputMessage objects
                            output_messages = []
                            for msg in parsed:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "assistant")
                                    parts = msg.get("parts", [])
                                    if parts and isinstance(parts, list):
                                        content = (
                                            parts[0].get("content", "")
                                            if isinstance(parts[0], dict)
                                            else str(parts[0])
                                        )
                                    else:
                                        content = msg.get("content", str(msg))
                                    finish_reason = msg.get(
                                        "finish_reason", "stop"
                                    )
                                    output_messages.append(
                                        OutputMessage(
                                            role=role,
                                            parts=[
                                                Text(
                                                    content=content,
                                                    type="text",
                                                )
                                            ],
                                            finish_reason=finish_reason,
                                        )
                                    )
                    except json.JSONDecodeError:
                        # Plain text string - create single OutputMessage
                        output_messages = [
                            OutputMessage(
                                role="assistant",
                                parts=[
                                    Text(
                                        content=original_output_data,
                                        type="text",
                                    )
                                ],
                                finish_reason="stop",
                            )
                        ]
                        _logger.debug(
                            "[TL_PROCESSOR] Created OutputMessage from plain string: %s...",
                            original_output_data[:50],
                        )

            # Serialize to JSON and store as gen_ai.* attributes (for span export)
            if input_messages:
                # Convert to OTel format: list of dicts with role and parts
                input_json = json.dumps(
                    [
                        {
                            "role": msg.role,
                            "parts": [
                                {"type": "text", "content": part.content}
                                for part in msg.parts
                            ],
                        }
                        for msg in input_messages
                    ]
                )
                mutated_attrs["gen_ai.input.messages"] = input_json

            if output_messages:
                output_json = json.dumps(
                    [
                        {
                            "role": msg.role,
                            "parts": [
                                {"type": "text", "content": part.content}
                                for part in msg.parts
                            ],
                            "finish_reason": getattr(
                                msg, "finish_reason", "stop"
                            ),
                        }
                        for msg in output_messages
                    ]
                )
                mutated_attrs["gen_ai.output.messages"] = output_json

            _logger.debug(
                "[TL_PROCESSOR] Messages reconstructed in mutation: input=%d, output=%d, span=%s",
                len(input_messages) if input_messages else 0,
                len(output_messages) if output_messages else 0,
                span_name,
            )

            # Cache the Python message objects for later use (avoid second reconstruction)
            if span_id is not None:
                self._message_cache[span_id] = (
                    input_messages,
                    output_messages,
                )
                _logger.debug(
                    "[TL_PROCESSOR] Cached messages for span_id=%s: input=%d, output=%d",
                    span_id,
                    len(input_messages) if input_messages else 0,
                    len(output_messages) if output_messages else 0,
                )

            return (input_messages, output_messages)

        except Exception as e:
            _logger.debug(
                "[TL_PROCESSOR] Message reconstruction in mutation failed: %s, span=%s",
                e,
                span_name,
            )
            return None

    def _mutate_span_if_needed(self, span: ReadableSpan) -> None:
        """Mutate the original span's attributes and name if configured to do so.

        This should be called early in on_end() before other processors see the span.
        """
        # Check if this span should be transformed
        if not self.span_filter(span):
            return

        # Skip if already processed (original Traceloop spans)
        if span.attributes and "_traceloop_processed" in span.attributes:
            return

        # Skip synthetic spans we created (CRITICAL: prevents infinite recursion)
        if span.attributes and "_traceloop_translated" in span.attributes:
            return

        # Determine which transformation set to use
        applied_rule: Optional[TransformationRule] = None
        for rule in self.rules:
            try:
                if rule.matches(span):
                    applied_rule = rule
                    break
            except Exception as match_err:  # pragma: no cover - defensive
                logging.warning(
                    "[TL_PROCESSOR] Rule match error: %s", match_err
                )

        # Decide which transformation config to apply
        if applied_rule is not None:
            attr_tx = applied_rule.attribute_transformations
            name_tx = applied_rule.name_transformations
        else:
            attr_tx = self.attribute_transformations
            name_tx = self.name_transformations

        # Check if mutation is enabled (both processor-level and per-invocation level)
        # For now, we only check processor-level since we don't have the invocation yet
        should_mutate = self.mutate_original_span

        # Mutate attributes
        if should_mutate and attr_tx:
            try:
                _logger = logging.getLogger(__name__)
                if hasattr(span, "_attributes"):
                    original = (
                        dict(span._attributes) if span._attributes else {}
                    )  # type: ignore[attr-defined]
                    mutated = self._apply_attribute_transformations(
                        original.copy(), attr_tx
                    )

                    # CRITICAL: Only reconstruct messages for LLM operations (chat, completion, embedding)
                    # NOT for evaluation spans or other non-LLM spans
                    # Check gen_ai.operation.name (set during transformation) to determine if this is an LLM span
                    operation_name = mutated.get("gen_ai.operation.name", "")
                    # Check span_kind from both transformed and original attributes (fallback for safety)
                    span_kind = mutated.get(
                        "gen_ai.span.kind", ""
                    ) or original.get("traceloop.span.kind", "")

                    # Fallback: infer from span name if operation name not set
                    if not operation_name and span.name:
                        span_name_lower = span.name.lower()
                        for pattern in [
                            "openai.chat",
                            "anthropic.chat",
                            ".chat",
                            "chat ",
                            "completion",
                            "embed",
                        ]:
                            if pattern in span_name_lower:
                                operation_name = (
                                    "chat"
                                    if "chat" in pattern
                                    else (
                                        "embedding"
                                        if "embed" in pattern
                                        else "completion"
                                    )
                                )
                                _logger.debug(
                                    "[TL_PROCESSOR] Inferred operation from span name: %s → %s",
                                    span.name,
                                    operation_name,
                                )
                                break

                    # Set gen_ai.operation.name based on span_kind for agent/workflow/task spans
                    # This is CRITICAL for _build_invocation to create the correct invocation type
                    # NOTE: We check span_kind BEFORE relying on operation_name because transformations
                    # might set a default operation_name (e.g., "chat") that doesn't reflect the actual span type
                    if span_kind:
                        span_kind_lower = str(span_kind).lower()
                        if span_kind_lower == "workflow":
                            # Use invoke_workflow for workflows but it is not a standard OTel GenAI operation, yet.
                            operation_name = "invoke_workflow"
                            mutated["gen_ai.operation.name"] = operation_name
                            _logger.debug(
                                "[TL_PROCESSOR] Set operation name for workflow span: %s → %s",
                                span.name,
                                operation_name,
                            )
                        elif span_kind_lower == "agent":
                            operation_name = "invoke_agent"
                            mutated["gen_ai.operation.name"] = operation_name
                            _logger.debug(
                                "[TL_PROCESSOR] Set operation name for agent span: %s → %s",
                                span.name,
                                operation_name,
                            )
                    # Check for explicit agent attributes if span_kind missed it
                    elif (
                        mutated.get("gen_ai.agent.name")
                        or mutated.get("gen_ai.agent.id")
                        or original.get("gen_ai.agent.name")
                        or original.get("gen_ai.agent.id")
                    ):
                        # Ensure we don't overwrite if it's already identified as something specific like chat
                        if (
                            not operation_name
                            or operation_name == "completion"
                            or operation_name == "unknown"
                        ):
                            operation_name = "invoke_agent"
                            mutated["gen_ai.operation.name"] = operation_name
                            _logger.debug(
                                "[TL_PROCESSOR] Set operation name for inferred agent span: %s → %s",
                                span.name,
                                operation_name,
                            )
                        else:
                            _logger.debug(
                                "[TL_PROCESSOR] Not setting invoke_agent because operation_name is already %s",
                                operation_name,
                            )

                    is_llm_operation = any(
                        op in str(operation_name).lower()
                        for op in ["chat", "completion", "embedding", "embed"]
                    )

                    # Treat Traceloop "agent" spans as invoke_agent operations
                    is_agent_operation = any(
                        op in str(operation_name).lower()
                        for op in ["invoke_agent", "create_agent"]
                    ) or any(
                        op in str(span_kind).lower()
                        # Removed workflow from here, handled separatedly
                        for op in ["agent"]
                    )

                    is_task_operation = any(
                        op in str(span_kind).lower() for op in ["task"]
                    )

                    if (
                        is_llm_operation
                        or is_agent_operation
                        or is_task_operation
                    ):
                        # This is an LLM span - reconstruct messages once and cache them
                        span_id = getattr(
                            getattr(span, "context", None), "span_id", None
                        )

                        self._reconstruct_and_set_messages(
                            original, mutated, span.name, span_id
                        )

                        # Note: Agent operations use structured input_messages/output_messages
                        # from the message cache directly in _build_invocation,
                        # so we don't need to populate additional string attributes here.

                        _logger.debug(
                            "[TL_PROCESSOR] Messages reconstructed for LLM span: operation=%s, span=%s, span_id=%s",
                            operation_name,
                            span.name,
                            span_id,
                        )
                    else:
                        # Not an LLM span - skip message reconstruction
                        _logger.debug(
                            "[TL_PROCESSOR] Skipping message reconstruction for non-LLM span: operation=%s, span=%s",
                            operation_name,
                            span.name,
                        )

                    # Mark as processed
                    mutated["_traceloop_processed"] = True
                    # Clear and update the underlying _attributes dict
                    span._attributes.clear()  # type: ignore[attr-defined]
                    span._attributes.update(mutated)  # type: ignore[attr-defined]
                    logging.getLogger(__name__).debug(
                        "Mutated span %s attributes: %s -> %s keys",
                        span.name,
                        len(original),
                        len(mutated),
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "Span %s does not have _attributes; mutation skipped",
                        span.name,
                    )
            except Exception as mut_err:
                logging.getLogger(__name__).debug(
                    "Attribute mutation skipped due to error: %s", mut_err
                )

        # Mutate name
        if should_mutate and name_tx:
            try:
                new_name = self._derive_new_name(span.name, name_tx)
                if new_name and hasattr(span, "_name"):
                    span._name = new_name  # type: ignore[attr-defined]
                    logging.getLogger(__name__).debug(
                        "Mutated span name: %s -> %s", span.name, new_name
                    )
                elif new_name and hasattr(span, "update_name"):
                    try:
                        span.update_name(new_name)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception as name_err:
                logging.getLogger(__name__).debug(
                    "Span name mutation failed: %s", name_err
                )

    def _apply_attribute_transformations(
        self, base: Dict[str, Any], transformations: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not transformations:
            return base
        remove_keys = transformations.get("remove") or []
        for k in remove_keys:
            base.pop(k, None)
        rename_map = transformations.get("rename") or {}
        for old, new in rename_map.items():
            if old in base:
                value = base.pop(old)
                # Special handling for entity input/output - normalize and serialize
                if old in (
                    "traceloop.entity.input",
                    "traceloop.entity.output",
                ):
                    try:
                        direction = "input" if "input" in old else "output"
                        normalized = normalize_traceloop_content(
                            value, direction
                        )
                        value = json.dumps(normalized)
                    except Exception as e:
                        # If normalization fails, try to serialize as-is
                        logging.getLogger(__name__).warning(
                            f"Failed to normalize {old}: {e}, using raw value"
                        )
                        try:
                            value = (
                                json.dumps(value)
                                if not isinstance(value, str)
                                else value
                            )
                        except Exception:
                            value = str(value)
                base[new] = value
        add_map = transformations.get("add") or {}
        for k, v in add_map.items():
            base[k] = v
        return base

    def _derive_new_name(
        self,
        original_name: str,
        name_transformations: Optional[Dict[str, str]],
    ) -> Optional[str]:
        if not name_transformations:
            return None

        for pattern, new_name in name_transformations.items():
            try:
                if fnmatch.fnmatch(original_name, pattern):
                    return new_name
            except Exception:
                continue
        return None

    def _convert_langchain_to_genai_messages(
        self, langchain_messages: Optional[List], direction: str
    ) -> List:
        """
        Convert LangChain messages to GenAI SDK message format.

        LangChain messages have .content directly, but GenAI SDK expects
        messages with .parts containing Text/ToolCall objects.
        """
        if not langchain_messages:
            return []

        genai_messages = []
        for lc_msg in langchain_messages:
            try:
                # Extract role from LangChain message type
                msg_type = type(lc_msg).__name__.lower()
                if "human" in msg_type or "user" in msg_type:
                    role = "user"
                elif "ai" in msg_type or "assistant" in msg_type:
                    role = "assistant"
                elif "system" in msg_type:
                    role = "system"
                elif "tool" in msg_type:
                    role = "tool"
                elif "function" in msg_type:
                    role = "function"
                else:
                    role = getattr(lc_msg, "role", "user")

                # Extract content and convert to parts
                content = getattr(lc_msg, "content", "")

                # CRITICAL 1: Check if content is a JSON string with LangChain serialization format
                # Basically only use the "content" of the incoming traceloop entity input/output
                if (
                    isinstance(content, str)
                    and content.startswith("{")
                    and '"lc"' in content
                ):
                    try:
                        parsed = json.loads(content)
                        # LangChain serialization format: {"lc": 1, "kwargs": {"content": "..."}}
                        if (
                            isinstance(parsed, dict)
                            and "kwargs" in parsed
                            and "content" in parsed["kwargs"]
                        ):
                            content = parsed["kwargs"]["content"]
                            logging.getLogger(__name__).debug(
                                "[TL_PROCESSOR] Extracted content from LangChain serialization format"
                            )
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logging.getLogger(__name__).warning(
                            "[TL_PROCESSOR] Failed to parse LangChain serialization: %s",
                            str(e),
                        )

                # CRITICAL 2: Ensure content is a string, not a dict or other object
                if isinstance(content, dict):
                    # If content is a dict, it might be already structured
                    # Try to extract the actual text from it
                    if "content" in content:
                        content = content["content"]
                    elif "parts" in content and isinstance(
                        content["parts"], list
                    ):
                        # Extract from parts structure
                        text_parts = [
                            p.get("content", "")
                            for p in content["parts"]
                            if isinstance(p, dict)
                        ]
                        content = " ".join(text_parts)
                    else:
                        # Fallback: serialize to JSON string (not ideal)
                        content = json.dumps(content)
                        logging.getLogger(__name__).warning(
                            "[TL_PROCESSOR] Content is dict, serializing: %s",
                            str(content)[:100],
                        )

                parts = [Text(content=str(content))] if content else []

                # Create GenAI SDK message
                if direction == "output":
                    finish_reason = getattr(lc_msg, "finish_reason", "stop")
                    genai_msg = OutputMessage(
                        role=role, parts=parts, finish_reason=finish_reason
                    )
                else:
                    genai_msg = InputMessage(role=role, parts=parts)

                genai_messages.append(genai_msg)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    f"Failed to convert LangChain message: {e}"
                )
                continue

        return genai_messages

    def _build_invocation(
        self,
        existing_span: ReadableSpan,
        *,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentCreation | AgentInvocation | Workflow | LLMInvocation]:
        base_attrs: Dict[str, Any] = (
            dict(existing_span.attributes) if existing_span.attributes else {}
        )

        # BEFORE transforming attributes, extract original message data
        # for message reconstruction (needed for evaluations)
        # Try multiple attribute names from different instrumentation sources:
        # 1. gen_ai.* (OTel GenAI format)
        # 2. traceloop.entity.* (Traceloop SDK)
        # 3. llm.* (older OpenAI instrumentation)
        # 4. gen_ai.content.* (another format variant)
        original_input_data = (
            base_attrs.get("gen_ai.input.messages")
            or base_attrs.get("gen_ai.input.message")
            or base_attrs.get("traceloop.entity.input")
            or base_attrs.get("gen_ai.prompt")
            or base_attrs.get("llm.prompts")
            or base_attrs.get("gen_ai.content.prompt")
        )
        original_output_data = (
            base_attrs.get("gen_ai.output.messages")
            or base_attrs.get("gen_ai.output.message")
            or base_attrs.get("traceloop.entity.output")
            or base_attrs.get("gen_ai.completion")
            or base_attrs.get("llm.completions")
            or base_attrs.get("gen_ai.content.completion")
        )

        # Apply attribute transformations
        base_attrs = self._apply_attribute_transformations(
            base_attrs, attribute_transformations
        )
        if traceloop_attributes:
            # Transform traceloop_attributes before adding them to avoid re-introducing removed keys
            transformed_tl_attrs = self._apply_attribute_transformations(
                traceloop_attributes.copy(), attribute_transformations
            )
            base_attrs.update(transformed_tl_attrs)

        # Final cleanup: remove any remaining traceloop.* keys that weren't in the rename map
        # This catches any attributes added by the Traceloop SDK or other sources
        keys_to_remove = [
            k for k in base_attrs.keys() if k.startswith("traceloop.")
        ]
        for k in keys_to_remove:
            base_attrs.pop(k, None)

        new_name = self._derive_new_name(
            existing_span.name, name_transformations
        )

        # Try to get model from various attribute sources
        request_model = (
            base_attrs.get("gen_ai.request.model")
            or base_attrs.get("gen_ai.response.model")
            or base_attrs.get("llm.request.model")
            or base_attrs.get("ai.model.name")
        )

        # Infer model from original span name pattern like "chat gpt-4" if not found
        if not request_model and existing_span.name:
            # Simple heuristic: take token(s) after first space
            parts = existing_span.name.strip().split()
            if len(parts) >= 2:
                candidate = parts[-1]  # Prefer last token (e.g., "gpt-4")
                # Basic sanity: exclude generic words that appear in indicators list
                if candidate.lower() not in {
                    "chat",
                    "completion",
                    "llm",
                    "ai",
                }:
                    request_model = candidate

        # For Traceloop task/workflow spans without model info, preserve original span name
        # instead of generating "chat unknown" or similar
        span_kind = base_attrs.get("gen_ai.span.kind") or base_attrs.get(
            "traceloop.span.kind"
        )
        if not request_model and span_kind in (
            "task",
            "workflow",
            "agent",
            "tool",
        ):
            # Use the original span name to avoid "chat unknown"
            if not new_name:
                new_name = existing_span.name
            request_model = "unknown"  # Still need a model for LLMInvocation
        elif not request_model:
            # Default to "unknown" only if we still don't have a model
            request_model = "unknown"

        # For spans that already have gen_ai.* attributes
        # preserve the original span name unless explicitly overridden
        if not new_name and base_attrs.get("gen_ai.system"):
            new_name = existing_span.name

        # Set the span name override if we have one
        if new_name:
            # Provide override for SpanEmitter (we extended it to honor this)
            base_attrs.setdefault("gen_ai.override.span_name", new_name)

        # Get messages from cache (reconstructed during mutation, no need to reconstruct again)
        span_id = getattr(
            getattr(existing_span, "context", None), "span_id", None
        )
        cached_messages = self._message_cache.get(span_id)

        _logger = logging.getLogger(__name__)
        _logger.debug(
            "[TL_PROCESSOR] _build_invocation: span_id=%s, cache_has_entry=%s, cache_size=%d, span=%s",
            span_id,
            span_id in self._message_cache if span_id else False,
            len(self._message_cache),
            existing_span.name,
        )

        if cached_messages:
            # Use cached messages (already in DeepEval format: InputMessage/OutputMessage with Text objects)
            input_messages, output_messages = cached_messages
            _logger.debug(
                "[TL_PROCESSOR] Using cached messages for invocation: input=%d, output=%d, span=%s, span_id=%s",
                len(input_messages) if input_messages else 0,
                len(output_messages) if output_messages else 0,
                existing_span.name,
                span_id,
            )
        else:
            # Fallback: try to reconstruct if not in cache (shouldn't happen for LLM spans)
            input_messages = None
            output_messages = None

            _logger.warning(
                "[TL_PROCESSOR] Messages NOT in cache! span_id=%s, span=%s, "
                "has_input_data=%s, has_output_data=%s, attr_keys=%s",
                span_id,
                existing_span.name,
                original_input_data is not None,
                original_output_data is not None,
                list(base_attrs.keys())[:25],
            )

            if original_input_data or original_output_data:
                try:
                    _logger.debug(
                        "[TL_PROCESSOR] Attempting fallback reconstruction: input_len=%d, output_len=%d",
                        len(str(original_input_data))
                        if original_input_data
                        else 0,
                        len(str(original_output_data))
                        if original_output_data
                        else 0,
                    )

                    lc_input, lc_output = reconstruct_messages_from_traceloop(
                        original_input_data, original_output_data
                    )
                    # Convert LangChain messages to GenAI SDK format for evaluations
                    input_messages = self._convert_langchain_to_genai_messages(
                        lc_input, "input"
                    )
                    output_messages = (
                        self._convert_langchain_to_genai_messages(
                            lc_output, "output"
                        )
                    )
                    _logger.debug(
                        "[TL_PROCESSOR] Fallback: reconstructed messages for invocation: input=%d, output=%d, span=%s",
                        len(input_messages) if input_messages else 0,
                        len(output_messages) if output_messages else 0,
                        existing_span.name,
                    )
                except Exception as e:
                    _logger.warning(
                        "[TL_PROCESSOR] Message reconstruction failed: %s, span=%s",
                        e,
                        existing_span.name,
                    )
            else:
                _logger.debug(
                    "[TL_PROCESSOR] ERROR: No message data available! span_id=%s, span=%s, attrs_keys=%s",
                    span_id,
                    existing_span.name,
                    list(base_attrs.keys())[:20],
                )
                # Log specific attribute values for debugging
                _logger.debug(
                    "[TL_PROCESSOR] Attribute values: gen_ai.input.messages=%s, traceloop.entity.input=%s, llm.prompts=%s",
                    base_attrs.get("gen_ai.input.messages", "MISSING")[:100]
                    if base_attrs.get("gen_ai.input.messages")
                    else "MISSING",
                    base_attrs.get("traceloop.entity.input", "MISSING")[:100]
                    if base_attrs.get("traceloop.entity.input")
                    else "MISSING",
                    base_attrs.get("llm.prompts", "MISSING")[:100]
                    if base_attrs.get("llm.prompts")
                    else "MISSING",
                )

        # Create invocation with reconstructed messages
        _logger = logging.getLogger(__name__)
        _logger.debug(
            "[TL_PROCESSOR] Creating invocation: input_msgs=%d, output_msgs=%d, span=%s, span_id=%s",
            len(input_messages) if input_messages else 0,
            len(output_messages) if output_messages else 0,
            existing_span.name,
            span_id,
        )

        # Determine invocation type based on operation name
        operation_name = base_attrs.get("gen_ai.operation.name", "")
        span_kind = base_attrs.get("gen_ai.span.kind") or base_attrs.get(
            "traceloop.span.kind"
        )

        # Determine invocation type based on operation name
        if operation_name == "create_agent":
            # Create AgentCreation invocation
            invocation = AgentCreation(
                name=base_attrs.get("gen_ai.agent.name") or existing_span.name,
                agent_type=base_attrs.get("gen_ai.agent.type") or None,
                description=base_attrs.get("gen_ai.agent.description"),
                model=request_model,
                framework=base_attrs.get("gen_ai.framework"),
                attributes=base_attrs,
            )
            # Extract tools if available
            tools_str = base_attrs.get("gen_ai.agent.tools")
            if tools_str:
                try:
                    invocation.tools = (
                        json.loads(tools_str)
                        if isinstance(tools_str, str)
                        else tools_str
                    )
                except Exception:
                    pass
            # Extract system instructions
            invocation.system_instructions = (
                base_attrs.get("gen_ai.system.instructions") or None
            )
            # Extract input from reconstructed messages or build from attributes
            if input_messages:
                invocation.input_messages = input_messages
            elif not invocation.input_messages:
                # Fallback: try to extract from span attributes and wrap in InputMessage
                fallback_input = (
                    base_attrs.get("input_context")
                    or base_attrs.get("input")
                    or base_attrs.get("initial_input")
                )
                if fallback_input:
                    invocation.input_messages = [
                        InputMessage(
                            role="user", parts=[Text(content=fallback_input)]
                        )
                    ]
            return invocation

        elif operation_name == "invoke_agent":
            # Create AgentInvocation
            invocation = AgentInvocation(
                name=base_attrs.get("gen_ai.agent.name") or existing_span.name,
                agent_type=base_attrs.get("gen_ai.agent.type") or None,
                description=base_attrs.get("gen_ai.agent.description"),
                model=request_model,
                framework=base_attrs.get("gen_ai.framework"),
                attributes=base_attrs,
            )
            # Extract tools if available
            tools_str = base_attrs.get("gen_ai.agent.tools")
            if tools_str:
                try:
                    invocation.tools = (
                        json.loads(tools_str)
                        if isinstance(tools_str, str)
                        else tools_str
                    )
                except Exception:
                    pass
            # Extract system instructions
            invocation.system_instructions = (
                base_attrs.get("gen_ai.system.instructions") or None
            )

            # Extract input from reconstructed messages or build from attributes
            if input_messages:
                invocation.input_messages = input_messages
            elif not invocation.input_messages:
                # Fallback: try to extract from span attributes and wrap in InputMessage
                fallback_input = (
                    base_attrs.get("input_context")
                    or base_attrs.get("input")
                    or base_attrs.get("initial_input")
                    or base_attrs.get("prompt")
                    or base_attrs.get("query")
                )
                # Secondary fallback: use original untransformed data (e.g. traceloop.entity.input)
                # This is critical when attributes were stripped and message reconstruction failed
                if not fallback_input and original_input_data:
                    if isinstance(original_input_data, (dict, list)):
                        fallback_input = json.dumps(original_input_data)
                    else:
                        fallback_input = str(original_input_data)
                if fallback_input:
                    invocation.input_messages = [
                        InputMessage(
                            role="user", parts=[Text(content=fallback_input)]
                        )
                    ]

            # Extract output from reconstructed messages or build from attributes
            if output_messages:
                invocation.output_messages = output_messages
            elif not invocation.output_messages:
                # Fallback: try to extract from span attributes and wrap in OutputMessage
                fallback_output = (
                    base_attrs.get("output_result")
                    or base_attrs.get("output")
                    or base_attrs.get("final_output")
                    or base_attrs.get("response")
                    or base_attrs.get("answer")
                )
                # Secondary fallback: use original untransformed data (e.g. traceloop.entity.output)
                if not fallback_output and original_output_data:
                    if isinstance(original_output_data, (dict, list)):
                        fallback_output = json.dumps(original_output_data)
                    else:
                        fallback_output = str(original_output_data)
                if fallback_output:
                    invocation.output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=fallback_output)],
                        )
                    ]

            # Skip if no input/output available for evaluation
            if (
                not invocation.input_messages
                and not invocation.output_messages
            ):
                _logger.warning(
                    "[TL_PROCESSOR] Skipping AgentInvocation - no input/output available! "
                    "span=%s, span_id=%s",
                    existing_span.name,
                    span_id,
                )
                return None
            return invocation
        else:
            # Create LLMInvocation (default for chat, completion, embedding)
            # CRITICAL: LLM invocations require messages for evaluation
            if not input_messages or not output_messages:
                _logger.warning(
                    "[TL_PROCESSOR] Skipping LLM invocation creation - no messages available! "
                    "span=%s, span_id=%s",
                    existing_span.name,
                    span_id,
                )
                return None

            if output_messages and all(
                not msg.parts for msg in output_messages
            ):
                _logger.warning(
                    "[TL_PROCESSOR] Skipping invocation creation - output messages have empty parts! "
                    "span=%s, span_id=%s, output_messages=%s",
                    existing_span.name,
                    span_id,
                    output_messages,
                )
                return None

            invocation = LLMInvocation(
                request_model=str(request_model),
                attributes=base_attrs,
                input_messages=input_messages or [],
                output_messages=output_messages or [],
            )
            # Mark operation heuristically from original span name or operation attribute
            if operation_name:
                if "embed" in operation_name.lower():
                    invocation.operation = "embedding"  # type: ignore[attr-defined]
                elif "chat" in operation_name.lower():
                    invocation.operation = "chat"  # type: ignore[attr-defined]
                elif "completion" in operation_name.lower():
                    invocation.operation = "completion"  # type: ignore[attr-defined]
            else:
                # Fallback to inferring from span name
                lowered = existing_span.name.lower()
                if lowered.startswith("embed"):
                    invocation.operation = "embedding"  # type: ignore[attr-defined]
                elif lowered.startswith("chat"):
                    invocation.operation = "chat"  # type: ignore[attr-defined]
            return invocation
