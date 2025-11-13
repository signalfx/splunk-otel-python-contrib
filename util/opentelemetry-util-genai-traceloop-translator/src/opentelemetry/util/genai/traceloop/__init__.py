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

import logging
import os
from typing import Any, Dict

from opentelemetry import trace

_ENV_DISABLE = "OTEL_INSTRUMENTATION_GENAI_TRACELOOP_DISABLE"
_LOGGER = logging.getLogger(__name__)

# Default attribute transformation mappings (traceloop.* -> gen_ai.*)
# This is the single source of truth for all attribute mappings.
#
# Mapping Status Legend:
# ✅ OFFICIAL: Documented in OpenTelemetry GenAI semantic conventions
# ⚠️  CUSTOM: Traceloop-specific, not in official semconv (extension attributes)
_DEFAULT_ATTR_TRANSFORMATIONS = {
    "rename": {
        # Content mappings (special handling with normalization)
        "traceloop.entity.input": "gen_ai.input.messages",  # OFFICIAL
        "traceloop.entity.output": "gen_ai.output.messages",  # OFFICIAL
        # Agent and conversation
        "traceloop.entity.name": "gen_ai.agent.name",  # OFFICIAL (agent spans)
        "traceloop.correlation.id": "gen_ai.conversation.id",  # OFFICIAL (Conditionally Required)
        # ⚠️  CUSTOM: Workflow / entity hierarchy (Traceloop-specific extensions)
        "traceloop.workflow.name": "gen_ai.workflow.name",
        "traceloop.entity.path": "gen_ai.workflow.path",
        "traceloop.association.properties": "gen_ai.association.properties",
        "traceloop.entity.version": "gen_ai.workflow.version",
        "traceloop.span.kind": "gen_ai.span.kind",
        "llm.request.type": "gen_ai.operation.name",
        "gen_ai.completion.0.content": "gen_ai.output.messages",
        "gen_ai.prompt.0.content": "gen_ai.input.messages",
    }
}

# Default span name transformation mappings
_DEFAULT_NAME_TRANSFORMATIONS = {"chat *": "genai.chat"}

# Global flag to track if processor has been registered (prevents multiple instances)
_PROCESSOR_REGISTERED = False


def enable_traceloop_translator(
    *,
    attribute_transformations: Dict[str, Any] | None = None,
    name_transformations: Dict[str, str] | None = None,
    mutate_original_span: bool = True,
) -> bool:
    """Enable the Traceloop span translator processor.

    This function registers the TraceloopSpanProcessor with the global tracer provider.
    It's safe to call multiple times (idempotent).

    Args:
        attribute_transformations: Custom attribute transformation rules.
            If None, uses default transformations (traceloop.* -> gen_ai.*).
        name_transformations: Custom span name transformation rules.
            If None, uses default transformations (chat * -> genai.chat).
        mutate_original_span: If True, mutate the original span's attributes.
            If False, only create new synthetic spans.

    Returns:
        True if the processor was registered, False if already registered or disabled.
    """
    # CRITICAL: Check global flag first to prevent multiple processor instances
    global _PROCESSOR_REGISTERED
    if _PROCESSOR_REGISTERED:
        _LOGGER.debug(
            "TraceloopSpanProcessor already registered (global flag); skipping duplicate"
        )
        return False

    # Import here to avoid circular imports
    from ..processor.traceloop_span_processor import TraceloopSpanProcessor

    provider = trace.get_tracer_provider()

    # Check if provider supports span processors
    if not hasattr(provider, "add_span_processor"):
        _LOGGER.warning(
            "Tracer provider does not support span processors. "
            "TraceloopSpanProcessor cannot be registered. "
            "Make sure you're using the OpenTelemetry SDK TracerProvider."
        )
        return False

    # Check for existing processor to avoid duplicates
    for attr_name in ("_active_span_processors", "_span_processors"):
        existing = getattr(provider, attr_name, [])
        if isinstance(existing, (list, tuple)):
            for proc in existing:
                if isinstance(proc, TraceloopSpanProcessor):
                    _LOGGER.debug(
                        "TraceloopSpanProcessor already registered; skipping duplicate"
                    )
                    return False

    try:
        processor = TraceloopSpanProcessor(
            attribute_transformations=attribute_transformations
            or _DEFAULT_ATTR_TRANSFORMATIONS,
            name_transformations=name_transformations
            or _DEFAULT_NAME_TRANSFORMATIONS,
            mutate_original_span=mutate_original_span,
        )
        provider.add_span_processor(processor)
        _PROCESSOR_REGISTERED = True  # Set global flag to prevent duplicates
        _LOGGER.info(
            "TraceloopSpanProcessor registered automatically "
            "(disable with %s=true)",
            _ENV_DISABLE,
        )
        return True
    except (TypeError, ValueError) as config_err:
        # Fail-fast
        _LOGGER.error(
            "Invalid configuration for TraceloopSpanProcessor: %s",
            config_err,
            exc_info=True,
        )
        raise
    except Exception as exc:
        _LOGGER.warning(
            "Failed to register TraceloopSpanProcessor: %s", exc, exc_info=True
        )
        return False


def _auto_enable() -> None:
    """Automatically enable the translator unless explicitly disabled.

    This uses a deferred registration approach that works even if called before
    the TracerProvider is set up. It hooks into the OpenTelemetry trace module
    to register the processor as soon as a real TracerProvider is available.
    """
    if os.getenv(_ENV_DISABLE, "").lower() in {"1", "true", "yes", "on"}:
        _LOGGER.debug(
            "TraceloopSpanProcessor auto-registration skipped (disabled via %s)",
            _ENV_DISABLE,
        )
        return

    # Try immediate registration first
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        # Real provider exists - register immediately
        enable_traceloop_translator()
    else:
        _LOGGER.debug(
            "TracerProvider not ready yet; deferring TraceloopSpanProcessor registration"
        )
        _install_deferred_registration()


def _install_deferred_registration() -> None:
    """Install a hook to register the processor when TracerProvider becomes available."""
    from ..processor.traceloop_span_processor import TraceloopSpanProcessor

    # Wrap the trace.set_tracer_provider function to intercept when it's called
    original_set_tracer_provider = trace.set_tracer_provider

    def wrapped_set_tracer_provider(tracer_provider):
        """Wrapped version that auto-registers our processor."""
        # Call the original first
        result = original_set_tracer_provider(tracer_provider)

        # Now try to register our processor
        try:
            if hasattr(tracer_provider, "add_span_processor"):
                # Check if already registered to avoid duplicates
                already_registered = False
                for attr_name in (
                    "_active_span_processors",
                    "_span_processors",
                ):
                    existing = getattr(tracer_provider, attr_name, [])
                    if isinstance(existing, (list, tuple)):
                        for proc in existing:
                            if isinstance(proc, TraceloopSpanProcessor):
                                already_registered = True
                                break
                    if already_registered:
                        break

                if not already_registered:
                    # Double-check global flag before registering
                    global _PROCESSOR_REGISTERED
                    if _PROCESSOR_REGISTERED:
                        _LOGGER.debug(
                            "TraceloopSpanProcessor already registered (global flag); skipping deferred registration"
                        )
                        return result

                    processor = TraceloopSpanProcessor(
                        attribute_transformations=_DEFAULT_ATTR_TRANSFORMATIONS,
                        name_transformations=_DEFAULT_NAME_TRANSFORMATIONS,
                        mutate_original_span=True,
                    )
                    tracer_provider.add_span_processor(processor)
                    _PROCESSOR_REGISTERED = True  # Set global flag
                    _LOGGER.info(
                        "TraceloopSpanProcessor registered (deferred) after TracerProvider setup"
                    )
        except Exception as exc:
            _LOGGER.debug(
                "Failed to auto-register TraceloopSpanProcessor: %s", exc
            )

        return result

    # Install the wrapper
    trace.set_tracer_provider = wrapped_set_tracer_provider


# Auto-enable on import (unless disabled)
_auto_enable()


__all__ = [
    "enable_traceloop_translator",
]
