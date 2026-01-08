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

_ENV_DISABLE = "OTEL_INSTRUMENTATION_GENAI_openlit_DISABLE"
_LOGGER = logging.getLogger(__name__)

# Marker attribute to identify our wrapper (for conflict detection)
_WRAPPER_MARKER = "_openlit_translator_wrapper"

# Default attribute transformation mappings i.e., openlit specific ones to GenAI semantic convention
#
# These mappings translate OpenLit-specific attributes (including those marked as "Extra"
# in their semconv) to their OpenTelemetry GenAI semantic convention compliant equivalents.
#
# Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
#
_DEFAULT_ATTR_TRANSFORMATIONS = {
    "rename": {
        # OpenLit uses indexed content format, OTel uses structured messages
        "gen_ai.prompt": "gen_ai.input.messages",
        "gen_ai.completion": "gen_ai.output.messages",
        "gen_ai.content.prompt": "gen_ai.input.messages",
        "gen_ai.content.completion": "gen_ai.output.messages",
        #  GenAI Request Attributes (Extra) -> OTel semconv
        "gen_ai.request.embedding_dimension": "gen_ai.embeddings.dimension.count",
        #  GenAI Token Usage (Extra - alternative naming) -> OTel semconv
        "gen_ai.token.usage.input": "gen_ai.usage.input_tokens",
        "gen_ai.token.usage.output": "gen_ai.usage.output_tokens",
        #  GenAI LLM Provider Attributes (Extra - nested namespace) -> OTel semconv
        "gen_ai.llm.provider": "gen_ai.system",
        "gen_ai.llm.model": "gen_ai.request.model",
        "gen_ai.llm.temperature": "gen_ai.request.temperature",
        "gen_ai.llm.max_tokens": "gen_ai.request.max_tokens",
        "gen_ai.llm.top_p": "gen_ai.request.top_p",
        #  GenAI Operation Type (Extra) -> OTel semconv
        "gen_ai.operation.type": "gen_ai.operation.name",
        #  GenAI Output Messages (Extra - alternative naming) -> OTel semconv
        "gen_ai.output_messages": "gen_ai.output.messages",
        #  GenAI Session/Conversation Tracking (Extra) -> OTel semconv
        "gen_ai.session.id": "gen_ai.conversation.id",
        # OpenAI-specific Attributes -> OTel semconv
        "gen_ai.openai.thread.id": "gen_ai.conversation.id",
        #  GenAI Tool Attributes (Extra) -> OTel semconv
        # Normalize tool-related attributes to standard OTel tool attributes
        "gen_ai.tool.call.id": "gen_ai.tool.call.id",
        "gen_ai.tool.args": "gen_ai.tool.call.arguments",
        "gen_ai.tool.result": "gen_ai.tool.call.result",
        #  VectorDB Attributes (Extra) -> OTel DB semconv
        # Note: These map to OTel database semantic conventions, not gen_ai
        "gen_ai.vectordb.name": "db.system.name",
        "gen_ai.vectordb.search.query": "db.query.text",
        "gen_ai.vectordb.search.results_count": "db.response.returned_rows",
    }
}

# Default span name transformation mappings
_DEFAULT_NAME_TRANSFORMATIONS = {"chat *": "genai.chat"}

# Global flag to track if processor has been registered (prevents multiple instances)
_PROCESSOR_REGISTERED = False


def enable_openlit_translator(
    *,
    attribute_transformations: Dict[str, Any] | None = None,
    name_transformations: Dict[str, str] | None = None,
    mutate_original_span: bool = True,
) -> bool:
    """Enable the Openlit span translator processor.

    This function registers the OpenlitSpanProcessor with the global tracer provider.
    It's safe to call multiple times (idempotent).

    Args:
        attribute_transformations: Custom attribute transformation rules.
        name_transformations: Custom span name transformation rules.
        mutate_original_span: If True, mutate the original span's attributes.

    Returns:
        True if the processor was registered, False if already registered or disabled.
    """
    # CRITICAL: Check global flag first to prevent multiple processor instances
    global _PROCESSOR_REGISTERED
    if _PROCESSOR_REGISTERED:
        _LOGGER.debug(
            "OpenlitSpanProcessor already registered (global flag); skipping duplicate"
        )
        return False

    # Import here to avoid circular imports
    from ..processor.openlit_span_processor import OpenlitSpanProcessor

    provider = trace.get_tracer_provider()

    # Check if provider supports span processors
    if not hasattr(provider, "add_span_processor"):
        _LOGGER.warning(
            "Tracer provider does not support span processors. "
            "OpenlitSpanProcessor cannot be registered. "
            "Make sure you're using the OpenTelemetry SDK TracerProvider."
        )
        return False

    # Check for existing processor to avoid duplicates
    for attr_name in ("_active_span_processors", "_span_processors"):
        existing = getattr(provider, attr_name, [])
        if isinstance(existing, (list, tuple)):
            for proc in existing:
                if isinstance(proc, OpenlitSpanProcessor):
                    _LOGGER.debug(
                        "OpenlitSpanProcessor already registered; skipping duplicate"
                    )
                    return False

    try:
        processor = OpenlitSpanProcessor(
            attribute_transformations=attribute_transformations
            or _DEFAULT_ATTR_TRANSFORMATIONS,
            name_transformations=name_transformations
            or _DEFAULT_NAME_TRANSFORMATIONS,
            mutate_original_span=mutate_original_span,
        )
        provider.add_span_processor(processor)
        _PROCESSOR_REGISTERED = True  # Set global flag to prevent duplicates
        _LOGGER.info(
            "OpenlitSpanProcessor registered automatically "
            "(disable with %s=true)",
            _ENV_DISABLE,
        )
        return True
    except (TypeError, ValueError) as config_err:
        # Fail-fast
        _LOGGER.error(
            "Invalid configuration for OpenlitSpanProcessor: %s",
            config_err,
            exc_info=True,
        )
        raise
    except Exception as exc:
        _LOGGER.warning(
            "Failed to register OpenlitSpanProcessor: %s", exc, exc_info=True
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
            "OpenlitSpanProcessor auto-registration skipped (disabled via %s)",
            _ENV_DISABLE,
        )
        return

    # Try immediate registration first
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        # Real provider exists - register immediately
        enable_openlit_translator()
    else:
        _LOGGER.debug(
            "TracerProvider not ready yet; deferring OpenlitSpanProcessor registration"
        )
        _install_deferred_registration()


def _install_deferred_registration() -> None:
    """Install a hook to register the processor when TracerProvider becomes available."""
    from ..processor.openlit_span_processor import OpenlitSpanProcessor

    # Check if another translator has already wrapped set_tracer_provider
    current_func = trace.set_tracer_provider
    if hasattr(current_func, "_traceloop_translator_wrapper"):
        _LOGGER.info(
            "Traceloop translator is already installed; "
            "skipping OpenLit translator to avoid conflicts."
        )
        return

    # Check if we already wrapped it (prevent double-wrapping on re-import)
    if hasattr(current_func, _WRAPPER_MARKER):
        _LOGGER.debug("OpenLit translator wrapper already installed; skipping")
        return

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
                            if isinstance(proc, OpenlitSpanProcessor):
                                already_registered = True
                                break
                    if already_registered:
                        break

                if not already_registered:
                    # Double-check global flag before registering
                    global _PROCESSOR_REGISTERED
                    if _PROCESSOR_REGISTERED:
                        _LOGGER.debug(
                            "OpenlitSpanProcessor already registered (global flag); skipping deferred registration"
                        )
                        return result

                    processor = OpenlitSpanProcessor(
                        attribute_transformations=_DEFAULT_ATTR_TRANSFORMATIONS,
                        name_transformations=_DEFAULT_NAME_TRANSFORMATIONS,
                        mutate_original_span=True,
                    )
                    tracer_provider.add_span_processor(processor)
                    _PROCESSOR_REGISTERED = True  # Set global flag
                    _LOGGER.info(
                        "OpenlitSpanProcessor registered (deferred) after TracerProvider setup"
                    )
        except Exception as exc:
            _LOGGER.debug(
                "Failed to auto-register OpenlitSpanProcessor: %s", exc
            )

        return result

    # Mark the wrapper so we can detect it later
    wrapped_set_tracer_provider._openlit_translator_wrapper = True  # type: ignore[attr-defined]
    setattr(wrapped_set_tracer_provider, _WRAPPER_MARKER, True)

    # Install the wrapper
    trace.set_tracer_provider = wrapped_set_tracer_provider


# Auto-enable on import (unless disabled)
_auto_enable()


__all__ = [
    "enable_openlit_translator",
]
