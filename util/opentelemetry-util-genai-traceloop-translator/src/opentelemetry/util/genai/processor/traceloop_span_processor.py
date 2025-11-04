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
from opentelemetry.trace import Span

from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.genai.handler import (
    get_telemetry_handler,
    TelemetryHandler,
)

from .content_normalizer import normalize_traceloop_content

_ENV_RULES = "OTEL_GENAI_SPAN_TRANSFORM_RULES"


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
                logging.warning("Invalid regex in match_scope: %s", pattern)
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
            logging.warning("%s must contain a 'rules' list", _ENV_RULES)
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
        logging.warning("Failed to parse %s: %s", _ENV_RULES, exc)
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
        # Mapping from original span_id to translated INVOCATION (not span) for parent-child relationship preservation
        self._original_to_translated_invocation: Dict[int, Any] = {}

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
            "ai",
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

        # Skip spans we already produced (recursion guard) - check FIRST before span_filter
        if span.attributes and "_traceloop_processed" in span.attributes:
            return None

        # Check if this span should be transformed
        if not self.span_filter(span):
            logger.debug("Span %s filtered out by span_filter", span.name)
            return None

        logger.debug(
            "Processing span for transformation: %s (kind=%s)",
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
                logging.warning("Rule match error ignored: %s", match_err)

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
                        from opentelemetry.trace import set_span_in_context

                        parent_context = set_span_in_context(
                            translated_parent_span
                        )

            original_span_id = getattr(
                getattr(span, "context", None), "span_id", None
            )

            invocation.parent_context = parent_context
            handler.start_llm(invocation)
            # Set the sentinel attribute immediately on the new span to prevent recursion
            if invocation.span and invocation.span.is_recording():
                invocation.span.set_attribute("_traceloop_processed", True)
                # Store the mapping from original span_id to translated INVOCATION
                if original_span_id:
                    self._original_to_translated_invocation[
                        original_span_id
                    ] = invocation
            return invocation
        except Exception as emit_err:  # pragma: no cover - defensive
            logging.getLogger(__name__).warning(
                "Telemetry handler emission failed: %s", emit_err
            )
            return None

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span is ended. Mutate immediately, then process span translation.
        """
        try:
            # FIRST: Mutate the original span immediately (before other processors/exporters see it)
            # This ensures mutations happen before exporters capture the span
            self._mutate_span_if_needed(span)

            # THEN: Process span translation immediately for real-time telemetry
            # This ensures evaluations and other downstream processes work correctly
            result_invocation = self._process_span_translation(span)

            # Close the invocation immediately if one was created
            if result_invocation:
                handler = self.telemetry_handler or get_telemetry_handler()
                try:
                    handler.stop_llm(result_invocation)
                except Exception as stop_err:
                    logging.getLogger(__name__).warning(
                        "Failed to stop invocation: %s", stop_err
                    )

        except Exception as e:
            # Don't let transformation errors break the original span processing
            logging.warning(
                f"TraceloopSpanProcessor failed to transform span: {e}"
            )

    def shutdown(self) -> None:
        """Called when the tracer provider is shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True

    def _mutate_span_if_needed(self, span: ReadableSpan) -> None:
        """Mutate the original span's attributes and name if configured to do so.

        This should be called early in on_end() before other processors see the span.
        """
        # Check if this span should be transformed
        if not self.span_filter(span):
            return

        # Skip if already processed
        if span.attributes and "_traceloop_processed" in span.attributes:
            return

        # Determine which transformation set to use
        applied_rule: Optional[TransformationRule] = None
        for rule in self.rules:
            try:
                if rule.matches(span):
                    applied_rule = rule
                    break
            except Exception as match_err:  # pragma: no cover - defensive
                logging.warning("Rule match error ignored: %s", match_err)

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
                if hasattr(span, "_attributes"):
                    original = (
                        dict(span._attributes) if span._attributes else {}
                    )  # type: ignore[attr-defined]
                    mutated = self._apply_attribute_transformations(
                        original.copy(), attr_tx
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
        import fnmatch

        for pattern, new_name in name_transformations.items():
            try:
                if fnmatch.fnmatch(original_name, pattern):
                    return new_name
            except Exception:
                continue
        return None

    def _build_invocation(
        self,
        existing_span: ReadableSpan,
        *,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
    ) -> LLMInvocation:
        base_attrs: Dict[str, Any] = (
            dict(existing_span.attributes) if existing_span.attributes else {}
        )
        base_attrs = self._apply_attribute_transformations(
            base_attrs, attribute_transformations
        )
        if traceloop_attributes:
            # Transform traceloop_attributes before adding them to avoid re-introducing legacy keys
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
        invocation = LLMInvocation(
            request_model=str(request_model),
            attributes=base_attrs,
            # messages=[],
        )
        # Mark operation heuristically from original span name
        lowered = existing_span.name.lower()
        if lowered.startswith("embed"):
            invocation.operation = "embedding"  # type: ignore[attr-defined]
        elif lowered.startswith("chat"):
            invocation.operation = "chat"  # type: ignore[attr-defined]
        return invocation
