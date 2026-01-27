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
Langsmith Span Processor for translating Langsmith-specific attributes
to OpenTelemetry GenAI semantic convention compliant format.

Langsmith is the observability platform for LangChain applications. This
processor handles the conversion of Langsmith trace data to standardized
GenAI semantic conventions.

Reference: https://docs.smith.langchain.com/
"""

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
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
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

from .content_normalizer import normalize_langsmith_content
from .message_reconstructor import reconstruct_messages_from_langsmith

try:
    from opentelemetry.util.genai.version import __version__
except ImportError:
    __version__ = "0.0.0"

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
    langsmith_attributes: Dict[str, Any] = field(default_factory=dict)

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
                    "[LANGSMITH_PROCESSOR] Invalid regex in match_scope: %s",
                    pattern,
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
                "[LANGSMITH_PROCESSOR] %s must contain a 'rules' list",
                _ENV_RULES,
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
                    langsmith_attributes=r.get("langsmith_attributes", {})
                    or {},
                )
            )
        return rules
    except Exception as exc:  # broad: we never want to break app startup
        logging.warning(
            "[LANGSMITH_PROCESSOR] Failed to parse %s: %s", _ENV_RULES, exc
        )
        return []


class LangsmithSpanProcessor(SpanProcessor):
    """
    A span processor that automatically applies transformation rules to spans.

    This processor translates Langsmith-specific attributes to OpenTelemetry
    GenAI semantic convention compliant format. It can be added to your
    TracerProvider to automatically transform all spans according to your
    transformation rules.
    """

    def __init__(
        self,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        langsmith_attributes: Optional[Dict[str, Any]] = None,
        span_filter: Optional[Callable[[ReadableSpan], bool]] = None,
        rules: Optional[List[TransformationRule]] = None,
        load_env_rules: bool = True,
        telemetry_handler: Optional[TelemetryHandler] = None,
        mutate_original_span: bool = True,
    ):
        """
        Initialize the Langsmith span processor.

        Args:
            attribute_transformations: Rules for transforming span attributes
            name_transformations: Rules for transforming span names
            langsmith_attributes: Additional Langsmith-specific attributes to add
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
        self.langsmith_attributes = langsmith_attributes or {}
        self.span_filter = span_filter or self._default_span_filter
        # Load rule set (env + explicit). Explicit rules first for precedence.
        env_rules = _load_rules_from_env() if load_env_rules else []
        self.rules: List[TransformationRule] = list(rules or []) + env_rules
        self.telemetry_handler = telemetry_handler
        self.mutate_original_span = mutate_original_span
        if self.rules:
            logging.getLogger(__name__).debug(
                "LangsmithSpanProcessor loaded %d transformation rules (explicit=%d env=%d)",
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
        Langsmith task/workflow spans for transformation.
        """
        if not span.name:
            return False

        # Check for common LLM/AI span indicators
        llm_indicators = [
            "chat",
            "completion",
            "llm",
            "gpt",
            "claude",
            "gemini",
            "openai",
            "anthropic",
            "cohere",
            "huggingface",
            "langsmith",
            "langchain",
        ]

        span_name_lower = span.name.lower()
        for indicator in llm_indicators:
            if indicator in span_name_lower:
                return True

        # Check attributes for AI/LLM markers (if any attributes present)
        if span.attributes:
            # Check for langsmith metadata attributes
            for attr_key in span.attributes.keys():
                attr_key_lower = str(attr_key).lower()
                if any(
                    marker in attr_key_lower
                    for marker in ["llm", "ai", "gen_ai", "model", "langsmith"]
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
        # NOTE: _langsmith_processed is set by mutation, _langsmith_translated is set by translation
        if span.attributes and "_langsmith_translated" in span.attributes:
            return None

        # Check if this span should be transformed
        if not self.span_filter(span):
            logger.debug(
                "[LANGSMITH_PROCESSOR] Span filtered: name=%s", span.name
            )
            return None

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
                    "[LANGSMITH_PROCESSOR] Rule match error: %s", match_err
                )

        sentinel = {"_langsmith_processed": True}
        # Decide which transformation config to apply
        if applied_rule is not None:
            attr_tx = applied_rule.attribute_transformations
            name_tx = applied_rule.name_transformations
            extra_tl_attrs = {
                **applied_rule.langsmith_attributes,
                **sentinel,
            }
        else:
            attr_tx = self.attribute_transformations
            name_tx = self.name_transformations
            extra_tl_attrs = {**self.langsmith_attributes, **sentinel}

        # Build invocation (mutation already happened in on_end before this method)
        invocation = self._build_invocation(
            span,
            attribute_transformations=attr_tx,
            name_transformations=name_tx,
            langsmith_attributes=extra_tl_attrs,
        )

        # If invocation is None, it means we couldn't get messages - skip this span
        if invocation is None:
            logger.debug(
                "[LANGSMITH_PROCESSOR] Skipping span translation - invocation creation returned None: %s",
                span.name,
            )
            return None

        invocation.attributes.setdefault("_langsmith_processed", True)

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
                        "[LANGSMITH_PROCESSOR] Marked synthetic span ID=%s for skipping",
                        synthetic_span_id,
                    )

                # Also set attribute as defense-in-depth
                if (
                    hasattr(synthetic_span, "set_attribute")
                    and synthetic_span.is_recording()
                ):
                    try:
                        synthetic_span.set_attribute(
                            "_langsmith_translated", True
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
                "[LANGSMITH_PROCESSOR] Skipping synthetic span (ID in set): %s",
                span.name,
            )
            return True

        # Fallback: Also check attributes for defense-in-depth
        if span.attributes and "_langsmith_translated" in span.attributes:
            _logger.debug(
                "[LANGSMITH_PROCESSOR] Skipping synthetic span (attribute): %s",
                span.name,
            )
            return True

        # Skip already processed spans
        if span.attributes and "_langsmith_processed" in span.attributes:
            _logger.debug(
                "[LANGSMITH_PROCESSOR] Skipping already processed span: %s",
                span.name,
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
                        "[LANGSMITH_PROCESSOR] Span excluded (will not export): pattern='%s', span=%s",
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
                                "[LANGSMITH_PROCESSOR] Marked span as non-sampled: %s",
                                span_name,
                            )
                        except Exception as e:
                            _logger.debug(
                                "[LANGSMITH_PROCESSOR] Could not mark span as non-sampled: %s",
                                e,
                            )
                    return

            # STEP 2: Check if this is an LLM span that needs evaluation
            if self._is_llm_span(span):
                _logger.debug(
                    "[LANGSMITH_PROCESSOR] LLM span detected: %s, running evaluations on mutated span",
                    span.name,
                )

                invocation = self._build_invocation(
                    span,
                    attribute_transformations=self.attribute_transformations,
                    name_transformations=self.name_transformations,
                    langsmith_attributes=self.langsmith_attributes,
                )

                if invocation:
                    # Attach the original (mutated) span to the invocation
                    # This is normally done by start_llm, but we're skipping that
                    invocation.span = span  # type: ignore[attr-defined]

                    # Get the handler
                    handler = self.telemetry_handler or get_telemetry_handler()

                    # Extract trace context from the original span
                    span_context = getattr(span, "context", None)
                    trace_id = getattr(span_context, "trace_id", None)
                    span_id_val = getattr(span_context, "span_id", None)

                    # Set trace_id on invocation (needed for sampling)
                    invocation.trace_id = trace_id
                    invocation.span_id = span_id_val

                    # Set timing info (use span's timing if available)
                    # ReadableSpan has start_time and end_time in nanoseconds
                    if hasattr(span, "_start_time") and span._start_time:  # type: ignore[attr-defined]
                        invocation.start_time = (
                            span._start_time / 1e9
                        )  # Convert ns to seconds  # type: ignore[attr-defined]

                    # Use handler.finish() for full functionality
                    try:
                        handler.finish(invocation)
                        _logger.debug(
                            "[LANGSMITH_PROCESSOR] finish completed for span: %s, sampled=%s, trace_id=%s",
                            span.name,
                            invocation.sample_for_evaluation,
                            trace_id,
                        )

                        # If this invocation is an AgentInvocation, explicitly
                        # trigger agent-level evaluations
                        if isinstance(invocation, AgentInvocation):  # type: ignore[attr-defined]
                            try:
                                handler.evaluate_agent(invocation)
                                _logger.debug(
                                    "[LANGSMITH_PROCESSOR] Agent invocation evaluated: %s",
                                    span.name,
                                )
                            except (
                                Exception
                            ) as eval_err:  # pragma: no cover - defensive
                                _logger.warning(
                                    "[LANGSMITH_PROCESSOR] Failed to evaluate AgentInvocation: %s",
                                    eval_err,
                                )

                    except Exception as stop_err:
                        _logger.warning(
                            "[LANGSMITH_PROCESSOR] handler.finish failed: %s",
                            stop_err,
                        )
                else:
                    _logger.info(
                        "[LANGSMITH_PROCESSOR] Skipped evaluations (no invocation created): %s",
                        span.name,
                    )
            else:
                # Non-LLM spans (tasks, workflows, tools) - buffer for optional batch processing
                _logger.debug(
                    "[LANGSMITH_PROCESSOR] Non-LLM span buffered: %s (buffer_size=%d)",
                    span.name,
                    len(self._span_buffer) + 1,
                )
                self._span_buffer.append(span)

                # Process buffer when root span arrives (optional, for synthetic spans of workflows)
                if span.parent is None and not self._processing_buffer:
                    _logger.debug(
                        "[LANGSMITH_PROCESSOR] Root span detected, processing buffered spans (count=%d)",
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
                            _logger.debug(
                                "[LANGSMITH_PROCESSOR] Buffered span processed (mutation only): %s",
                                buffered_span.name,
                            )

                        self._span_buffer.clear()
                        self._original_to_translated_invocation.clear()
                    finally:
                        self._processing_buffer = False

        except Exception as e:
            # Don't let transformation errors break the original span processing
            logging.warning(
                "[LANGSMITH_PROCESSOR] Span transformation failed: %s", e
            )

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

        Simplified logic: Check if gen_ai.operation.name contains "chat" or other LLM operations.
        This is the most reliable way to identify actual LLM API calls vs orchestration spans.

        Returns True ONLY for actual LLM API call spans (gen_ai.operation.name = "chat", "completion", "embedding").
        Returns False for workflow orchestration, utility tasks, agent coordination, routing, etc.
        """
        _logger = logging.getLogger(__name__)

        if not span or not span.attributes:
            return False

        # Skip synthetic spans we already created (recursion guard)
        if span.attributes and "_langsmith_translated" in span.attributes:
            return False

        # CRITICAL: Exclude evaluation-related spans (prevent recursive evaluation)
        span_name = span.name or ""
        for exclude_pattern in _EXCLUDE_SPAN_PATTERNS:
            if exclude_pattern.lower() in span_name.lower():
                _logger.debug(
                    "[LANGSMITH_PROCESSOR] Span excluded (matches pattern '%s'): name=%s",
                    exclude_pattern,
                    span_name,
                )
                return False

        # ONLY CHECK: gen_ai.operation.name attribute (set during mutation in on_end)
        operation_name = span.attributes.get("gen_ai.operation.name")
        if operation_name:
            # Only trigger on actual LLM operations: chat, completion, embedding
            if any(
                op in str(operation_name).lower() for op in _LLM_OPERATIONS
            ):
                _logger.debug(
                    "[LANGSMITH_PROCESSOR] LLM span detected (gen_ai.operation.name=%s): name=%s",
                    operation_name,
                    span.name,
                )
                return True
            else:
                # Has operation name but not an LLM operation (e.g., "workflow", "task", "tool")
                _logger.debug(
                    "[LANGSMITH_PROCESSOR] Non-LLM operation (gen_ai.operation.name=%s): name=%s",
                    operation_name,
                    span.name,
                )
                return False

        # No gen_ai.operation.name means it wasn't transformed or doesn't match our rules
        _logger.debug(
            "[LANGSMITH_PROCESSOR] Span skipped (no gen_ai.operation.name): name=%s",
            span.name,
        )
        return False

    def _reconstruct_and_set_messages(
        self,
        original_attrs: dict,
        mutated_attrs: dict,
        span_name: str,
        span_id: Optional[int] = None,
    ) -> Optional[tuple]:
        """
        Reconstruct messages from Langsmith format and set them as gen_ai.* attributes.

        This ensures ALL spans have gen_ai.input.messages and gen_ai.output.messages
        in OTel format, not just spans processed for evaluation.

        Returns the reconstructed messages (input_messages, output_messages) for caching.
        """
        _logger = logging.getLogger(__name__)

        # Extract message data from various sources
        # 1. Already transformed: gen_ai.input.messages/output.messages
        # 2. Langsmith SDK format: langsmith.entity.input/output
        # 3. OpenAI instrumentation: gen_ai.prompt/gen_ai.completion (older format)
        # 4. LLM attributes: llm.prompts/llm.completions
        original_input_data = (
            mutated_attrs.get("gen_ai.input.messages")
            or original_attrs.get("langsmith.entity.input")
            or original_attrs.get("gen_ai.prompt")
            or original_attrs.get("llm.prompts")
            or original_attrs.get("gen_ai.content.prompt")
        )
        original_output_data = (
            mutated_attrs.get("gen_ai.output.messages")
            or original_attrs.get("langsmith.entity.output")
            or original_attrs.get("gen_ai.completion")
            or original_attrs.get("llm.completions")
            or original_attrs.get("gen_ai.content.completion")
        )

        if not original_input_data and not original_output_data:
            _logger.debug(
                "[LANGSMITH_PROCESSOR] No message data found in span attrs for reconstruction, "
                "available keys: %s",
                list(original_attrs.keys())[:15],
            )
            return None  # Nothing to reconstruct

        try:
            # First, try to reconstruct LangChain messages from Langsmith JSON format
            lc_input, lc_output = reconstruct_messages_from_langsmith(
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

            # FALLBACK: If LangChain reconstruction failed but we have plain string data,
            # create simple message objects directly.
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
                            "[LANGSMITH_PROCESSOR] Created InputMessage from plain string: %s...",
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
                                        # CRITICAL: Check if content is nested LangChain/generations JSON
                                        # and extract the actual message content
                                        if isinstance(
                                            content, str
                                        ) and content.startswith("{"):
                                            try:
                                                from .content_normalizer import (
                                                    _extract_langchain_messages,
                                                )

                                                extracted = _extract_langchain_messages(
                                                    content
                                                )
                                                if extracted:
                                                    # Use extracted message instead
                                                    ext_msg = extracted[0]
                                                    content = ext_msg.get(
                                                        "content", ""
                                                    )
                                                    role = ext_msg.get(
                                                        "role", role
                                                    )
                                                    # Get finish_reason and tool_calls from extracted
                                                    if (
                                                        "finish_reason"
                                                        in ext_msg
                                                    ):
                                                        msg[
                                                            "finish_reason"
                                                        ] = ext_msg[
                                                            "finish_reason"
                                                        ]
                                                    if "tool_calls" in ext_msg:
                                                        msg["tool_calls"] = (
                                                            ext_msg[
                                                                "tool_calls"
                                                            ]
                                                        )
                                            except Exception as e:
                                                _logger.debug(
                                                    "[LANGSMITH_PROCESSOR] Failed to extract nested content: %s",
                                                    e,
                                                )
                                    else:
                                        content = msg.get("content", str(msg))
                                    finish_reason = msg.get(
                                        "finish_reason", "stop"
                                    )
                                    # Build parts list - include tool_calls if present
                                    msg_parts = []
                                    if content:
                                        msg_parts.append(
                                            Text(content=content, type="text")
                                        )
                                    if msg.get("tool_calls"):
                                        # For now, represent tool calls as text (could be enhanced)
                                        for tc in msg["tool_calls"]:
                                            tc_text = f"Tool call: {tc.get('name', 'unknown')}"
                                            if tc.get("args"):
                                                tc_text += f"({json.dumps(tc['args'])})"
                                            msg_parts.append(
                                                Text(
                                                    content=tc_text,
                                                    type="text",
                                                )
                                            )
                                    if not msg_parts:
                                        # Empty content but might be tool call - add empty text
                                        msg_parts.append(
                                            Text(content="", type="text")
                                        )
                                    output_messages.append(
                                        OutputMessage(
                                            role=role,
                                            parts=msg_parts,
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
                            "[LANGSMITH_PROCESSOR] Created OutputMessage from plain string: %s...",
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
                "[LANGSMITH_PROCESSOR] Messages reconstructed in mutation: input=%d, output=%d, span=%s",
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
                    "[LANGSMITH_PROCESSOR] Cached messages for span_id=%s: input=%d, output=%d",
                    span_id,
                    len(input_messages) if input_messages else 0,
                    len(output_messages) if output_messages else 0,
                )

            return (input_messages, output_messages)

        except Exception as e:
            _logger.debug(
                "[LANGSMITH_PROCESSOR] Message reconstruction in mutation failed: %s, span=%s",
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

        # Skip if already processed (original langsmith spans)
        if span.attributes and "_langsmith_processed" in span.attributes:
            return

        # Skip synthetic spans we created (CRITICAL: prevents infinite recursion)
        if span.attributes and "_langsmith_translated" in span.attributes:
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
                    "[LANGSMITH_PROCESSOR] Rule match error: %s", match_err
                )

        # Decide which transformation config to apply
        if applied_rule is not None:
            attr_tx = applied_rule.attribute_transformations
            name_tx = applied_rule.name_transformations
            extra_attrs = applied_rule.langsmith_attributes
        else:
            attr_tx = self.attribute_transformations
            name_tx = self.name_transformations
            extra_attrs = self.langsmith_attributes

        # Check if mutation is enabled (both processor-level and per-invocation level)
        # For now, we only check processor-level since we don't have the invocation yet
        should_mutate = self.mutate_original_span

        # Mutate attributes
        if should_mutate and (attr_tx or extra_attrs):
            try:
                _logger = logging.getLogger(__name__)
                if hasattr(span, "_attributes"):
                    original = (
                        dict(span._attributes) if span._attributes else {}
                    )  # type: ignore[attr-defined]
                    mutated = self._apply_attribute_transformations(
                        original.copy(), attr_tx
                    )

                    # Apply extra langsmith attributes (e.g. gen_ai.system)
                    if extra_attrs:
                        mutated.update(extra_attrs)

                    # Check gen_ai.operation.name (set during transformation) to determine if this is an LLM span
                    operation_name = mutated.get("gen_ai.operation.name", "")
                    # Check span_kind from both transformed and original attributes (fallback for safety)
                    span_kind = mutated.get("gen_ai.span.kind", "")

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
                                    "[LANGSMITH_PROCESSOR] Inferred operation from span name: %s → %s",
                                    span.name,
                                    operation_name,
                                )
                                break

                    is_llm_operation = any(
                        op in str(operation_name).lower()
                        for op in ["chat", "completion", "embedding", "embed"]
                    )

                    is_agent_operation = any(
                        op in str(span_kind).lower() for op in ["agent"]
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
                        _logger.debug(
                            "[LANGSMITH_PROCESSOR] Messages reconstructed for LLM span: operation=%s, span=%s, span_id=%s",
                            operation_name,
                            span.name,
                            span_id,
                        )
                    else:
                        # Not an LLM span - skip message reconstruction
                        _logger.debug(
                            "[LANGSMITH_PROCESSOR] Skipping message reconstruction for non-LLM span: operation=%s, span=%s",
                            operation_name,
                            span.name,
                        )

                    # Mark as processed
                    mutated["_langsmith_processed"] = True
                    # Clear and update the underlying _attributes dict
                    span._attributes.clear()  # type: ignore[attr-defined]
                    span._attributes.update(mutated)  # type: ignore[attr-defined]

                    # CRITICAL: Mutate the instrumentation scope to match our handler
                    try:
                        new_scope = InstrumentationScope(
                            name="opentelemetry.util.genai.handler",
                            version=__version__,
                        )
                        span._instrumentation_scope = new_scope  # type: ignore[attr-defined]
                        _logger.debug(
                            "Mutated span %s instrumentation scope to: %s",
                            span.name,
                            new_scope.name,
                        )
                    except Exception as scope_err:
                        _logger.debug(
                            "Instrumentation scope mutation failed: %s",
                            scope_err,
                        )

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
                    "langsmith.entity.input",
                    "langsmith.entity.output",
                ):
                    try:
                        direction = "input" if "input" in old else "output"
                        normalized = normalize_langsmith_content(
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

        # Post-process: Normalize gen_ai.input/output.messages if they contain raw LangChain data
        # This handles the case where Langsmith native OTEL export sets these directly
        for msg_attr, direction in [
            ("gen_ai.input.messages", "input"),
            ("gen_ai.output.messages", "output"),
        ]:
            if msg_attr in base:
                value = base[msg_attr]
                # Check if it looks like it contains raw LangChain/generations data
                if isinstance(value, str) and (
                    '"generations"' in value or '"lc":' in value
                ):
                    try:
                        normalized = normalize_langsmith_content(
                            value, direction
                        )
                        base[msg_attr] = json.dumps(normalized)
                        logging.getLogger(__name__).debug(
                            f"Normalized existing {msg_attr} containing raw LangChain data"
                        )
                    except Exception as e:
                        logging.getLogger(__name__).debug(
                            f"Failed to normalize {msg_attr}: {e}"
                        )

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
                                "[LANGSMITH_PROCESSOR] Extracted content from LangChain serialization format"
                            )
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logging.getLogger(__name__).warning(
                            "[LANGSMITH_PROCESSOR] Failed to parse LangChain serialization: %s",
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
                            "[LANGSMITH_PROCESSOR] Content is dict, serializing: %s",
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
        langsmith_attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentCreation | AgentInvocation | Workflow | LLMInvocation]:
        # CRITICAL: Read from _attributes (the live/mutated dict), NOT from .attributes
        if hasattr(existing_span, "_attributes") and existing_span._attributes:  # type: ignore[attr-defined]
            base_attrs: Dict[str, Any] = dict(existing_span._attributes)  # type: ignore[attr-defined]
        else:
            base_attrs: Dict[str, Any] = (
                dict(existing_span.attributes)
                if existing_span.attributes
                else {}
            )

        # Check if span was already mutated (has _langsmith_processed marker)
        already_mutated = base_attrs.get("_langsmith_processed", False)

        # BEFORE transforming attributes, extract original message data
        original_input_data = (
            base_attrs.get("gen_ai.input.messages")
            or base_attrs.get("gen_ai.input.message")
            or base_attrs.get("langsmith.entity.input")
            or base_attrs.get("gen_ai.prompt")
            or base_attrs.get("llm.prompts")
            or base_attrs.get("gen_ai.content.prompt")
        )
        original_output_data = (
            base_attrs.get("gen_ai.output.messages")
            or base_attrs.get("gen_ai.output.message")
            or base_attrs.get("langsmith.entity.output")
            or base_attrs.get("gen_ai.completion")
            or base_attrs.get("llm.completions")
            or base_attrs.get("gen_ai.content.completion")
        )

        # Only apply attribute transformations if span was NOT already mutated
        if not already_mutated:
            base_attrs = self._apply_attribute_transformations(
                base_attrs, attribute_transformations
            )

        if langsmith_attributes:
            if not already_mutated:
                transformed_tl_attrs = self._apply_attribute_transformations(
                    langsmith_attributes.copy(), attribute_transformations
                )
                base_attrs.update(transformed_tl_attrs)
            else:
                base_attrs.update(langsmith_attributes)

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
            parts = existing_span.name.strip().split()
            if len(parts) >= 2:
                candidate = parts[-1]  # Prefer last token (e.g., "gpt-4")
                if candidate.lower() not in {
                    "chat",
                    "completion",
                    "llm",
                    "ai",
                }:
                    request_model = candidate

        # For langsmith task/workflow spans without model info, preserve original span name
        span_kind = base_attrs.get("gen_ai.span.kind")
        if not request_model and span_kind in (
            "task",
            "workflow",
            "agent",
            "tool",
        ):
            if not new_name:
                new_name = existing_span.name
            request_model = "unknown"
        elif not request_model:
            request_model = "unknown"

        if not new_name and base_attrs.get("gen_ai.system"):
            new_name = existing_span.name

        if new_name:
            base_attrs.setdefault("gen_ai.override.span_name", new_name)

        # Get messages from cache (reconstructed during mutation)
        span_id = getattr(
            getattr(existing_span, "context", None), "span_id", None
        )
        cached_messages = self._message_cache.get(span_id)

        _logger = logging.getLogger(__name__)
        _logger.debug(
            "[LANGSMITH_PROCESSOR] _build_invocation: span_id=%s, cache_has_entry=%s, cache_size=%d, span=%s",
            span_id,
            span_id in self._message_cache if span_id else False,
            len(self._message_cache),
            existing_span.name,
        )

        if cached_messages:
            input_messages, output_messages = cached_messages
            _logger.debug(
                "[LANGSMITH_PROCESSOR] Using cached messages for invocation: input=%d, output=%d, span=%s, span_id=%s",
                len(input_messages) if input_messages else 0,
                len(output_messages) if output_messages else 0,
                existing_span.name,
                span_id,
            )
        else:
            input_messages = None
            output_messages = None

            _logger.warning(
                "[LANGSMITH_PROCESSOR] Messages NOT in cache! span_id=%s, span=%s, has_input_data=%s, has_output_data=%s",
                span_id,
                existing_span.name,
                original_input_data is not None,
                original_output_data is not None,
            )

            if original_input_data or original_output_data:
                try:
                    _logger.debug(
                        "[LANGSMITH_PROCESSOR] Attempting fallback reconstruction: input_len=%d, output_len=%d",
                        len(str(original_input_data))
                        if original_input_data
                        else 0,
                        len(str(original_output_data))
                        if original_output_data
                        else 0,
                    )

                    lc_input, lc_output = reconstruct_messages_from_langsmith(
                        original_input_data, original_output_data
                    )
                    input_messages = self._convert_langchain_to_genai_messages(
                        lc_input, "input"
                    )
                    output_messages = (
                        self._convert_langchain_to_genai_messages(
                            lc_output, "output"
                        )
                    )
                    _logger.debug(
                        "[LANGSMITH_PROCESSOR] Fallback: reconstructed messages for invocation: input=%d, output=%d, span=%s",
                        len(input_messages) if input_messages else 0,
                        len(output_messages) if output_messages else 0,
                        existing_span.name,
                    )
                except Exception as e:
                    _logger.warning(
                        "[LANGSMITH_PROCESSOR] Message reconstruction failed: %s, span=%s",
                        e,
                        existing_span.name,
                    )
            else:
                _logger.error(
                    "[LANGSMITH_PROCESSOR] ERROR: No message data available! span_id=%s, span=%s, attrs_keys=%s",
                    span_id,
                    existing_span.name,
                    list(base_attrs.keys())[:20],
                )

        # Create invocation with reconstructed messages
        _logger.debug(
            "[LANGSMITH_PROCESSOR] Creating invocation: input_msgs=%d, output_msgs=%d, span=%s, span_id=%s",
            len(input_messages) if input_messages else 0,
            len(output_messages) if output_messages else 0,
            existing_span.name,
            span_id,
        )

        # Determine invocation type based on operation name
        operation_name = base_attrs.get("gen_ai.operation.name", "")

        if operation_name == "invoke_workflow":
            # Create Workflow invocation
            invocation = Workflow(
                name=base_attrs.get("gen_ai.workflow.name")
                or existing_span.name,
                workflow_type=base_attrs.get("gen_ai.workflow.type") or None,
                description=base_attrs.get("gen_ai.workflow.description"),
                framework=base_attrs.get("gen_ai.framework"),
                attributes=base_attrs,
            )
            if input_messages:
                invocation.initial_input = " ".join(
                    part.content
                    for msg in input_messages
                    for part in msg.parts
                    if hasattr(part, "content")
                )
            elif not invocation.initial_input:
                invocation.initial_input = (
                    base_attrs.get("initial_input")
                    or base_attrs.get("input")
                    or base_attrs.get("input_context")
                    or base_attrs.get("query")
                )

            if output_messages:
                invocation.final_output = " ".join(
                    part.content
                    for msg in output_messages
                    for part in msg.parts
                    if hasattr(part, "content")
                )
            elif not invocation.final_output:
                invocation.final_output = (
                    base_attrs.get("final_output")
                    or base_attrs.get("output")
                    or base_attrs.get("output_result")
                    or base_attrs.get("response")
                )
            return invocation

        elif operation_name == "create_agent":
            # Create AgentCreation invocation
            invocation = AgentCreation(
                name=base_attrs.get("gen_ai.agent.name") or existing_span.name,
                agent_type=base_attrs.get("gen_ai.agent.type") or None,
                description=base_attrs.get("gen_ai.agent.description"),
                model=request_model,
                framework=base_attrs.get("gen_ai.framework"),
                attributes=base_attrs,
            )
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
            invocation.system_instructions = (
                base_attrs.get("gen_ai.system.instructions") or None
            )
            if input_messages:
                invocation.input_context = " ".join(
                    part.content
                    for msg in input_messages
                    for part in msg.parts
                    if hasattr(part, "content")
                )
            elif not invocation.input_context:
                invocation.input_context = (
                    base_attrs.get("input_context")
                    or base_attrs.get("input")
                    or base_attrs.get("initial_input")
                )
            return invocation

        elif operation_name == "invoke_agent":
            # Create AgentInvocation
            invocation = AgentInvocation(
                name=base_attrs.get("gen_ai.agent.name") or existing_span.name,
                agent_type=base_attrs.get("gen_ai.agent.type"),
                description=base_attrs.get("gen_ai.agent.description"),
                model=request_model,
                framework=base_attrs.get("gen_ai.framework"),
                attributes=base_attrs,
            )
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
            invocation.system_instructions = (
                base_attrs.get("gen_ai.system.instructions") or None
            )

            if input_messages:
                invocation.input_context = " ".join(
                    part.content
                    for msg in input_messages
                    for part in msg.parts
                    if hasattr(part, "content")
                )
            elif not invocation.input_context:
                invocation.input_context = (
                    base_attrs.get("input_context")
                    or base_attrs.get("input")
                    or base_attrs.get("initial_input")
                    or base_attrs.get("prompt")
                    or base_attrs.get("query")
                )

            if output_messages:
                invocation.output_result = " ".join(
                    part.content
                    for msg in output_messages
                    for part in msg.parts
                    if hasattr(part, "content")
                )
            elif not invocation.output_result:
                invocation.output_result = (
                    base_attrs.get("output_result")
                    or base_attrs.get("output")
                    or base_attrs.get("final_output")
                    or base_attrs.get("response")
                    or base_attrs.get("answer")
                )

            if not invocation.input_context and not invocation.output_result:
                _logger.warning(
                    "[LANGSMITH_PROCESSOR] Skipping AgentInvocation - no input/output available! "
                    "span=%s, span_id=%s",
                    existing_span.name,
                    span_id,
                )
                return None
            return invocation
        else:
            # Create LLMInvocation (default for chat, completion, embedding)
            if not input_messages or not output_messages:
                _logger.warning(
                    "[LANGSMITH_PROCESSOR] Skipping LLM invocation creation - no messages available! "
                    "span=%s, span_id=%s",
                    existing_span.name,
                    span_id,
                )
                return None

            if output_messages and all(
                not msg.parts for msg in output_messages
            ):
                _logger.warning(
                    "[LANGSMITH_PROCESSOR] Skipping invocation creation - output messages have empty parts! "
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
            if operation_name:
                if "embed" in operation_name.lower():
                    invocation.operation = "embedding"  # type: ignore[attr-defined]
                elif "chat" in operation_name.lower():
                    invocation.operation = "chat"  # type: ignore[attr-defined]
                elif "completion" in operation_name.lower():
                    invocation.operation = "completion"  # type: ignore[attr-defined]
            else:
                lowered = existing_span.name.lower()
                if lowered.startswith("embed"):
                    invocation.operation = "embedding"  # type: ignore[attr-defined]
                elif lowered.startswith("chat"):
                    invocation.operation = "chat"  # type: ignore[attr-defined]
            return invocation
