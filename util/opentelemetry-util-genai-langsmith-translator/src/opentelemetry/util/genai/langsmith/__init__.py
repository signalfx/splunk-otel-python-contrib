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
Langsmith to GenAI Semantic Convention Translator.

This module provides automatic translation of Langsmith-specific span attributes
to OpenTelemetry GenAI semantic convention compliant format.

Langsmith is the observability platform for LangChain applications, providing
tracing, evaluation, and monitoring capabilities. This translator bridges
Langsmith's attribute format to the standardized GenAI semantic conventions.

Reference:
- Langsmith: https://docs.smith.langchain.com/
- GenAI Semconv: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from opentelemetry import trace

_ENV_DISABLE = "OTEL_INSTRUMENTATION_GENAI_LANGSMITH_DISABLE"
_LOGGER = logging.getLogger(__name__)

# Marker attribute to identify our wrapper (for conflict detection)
_WRAPPER_MARKER = "_langsmith_translator_wrapper"

# Default attribute transformation mappings i.e., Langsmith specific ones to GenAI semantic convention
#
# These mappings translate Langsmith-specific attributes to their OpenTelemetry
# GenAI semantic convention compliant equivalents.
#
# Langsmith traces LangChain applications and records metadata in its own format.
# This includes run metadata, model configurations, and I/O content.
#
# Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
#
_DEFAULT_ATTR_TRANSFORMATIONS = {
    "rename": {
        # --- 1. Content & Messages (Blob to Structured) ---
        # ✅ OFFICIAL: Input/Output message mappings
        "gen_ai.prompt": "gen_ai.input.messages",
        "gen_ai.content.prompt": "gen_ai.input.messages",
        "gen_ai.prompt.0.content": "gen_ai.input.messages",
        "gen_ai.completion": "gen_ai.output.messages",
        "gen_ai.content.completion": "gen_ai.output.messages",
        "gen_ai.completion.0.content": "gen_ai.output.messages",
        "gen_ai.output_messages": "gen_ai.output.messages",
        # ⚠️  CUSTOM: Langsmith entity input/output
        "langsmith.entity.input": "gen_ai.input.messages",
        "langsmith.entity.output": "gen_ai.output.messages",
        # --- 2. Core Model & System Attributes ---
        # ✅ OFFICIAL: Model and system identification
        "langsmith.metadata.ls_provider": "gen_ai.system",
        "langsmith.metadata.ls_model_name": "gen_ai.request.model",
        "langsmith.metadata.ls_model_type": "gen_ai.operation.name",
        # ⚠️  CUSTOM: Alternative Langsmith model attributes
        "langsmith.model_name": "gen_ai.request.model",
        "langsmith.provider": "gen_ai.system",
        "langsmith.run_type": "gen_ai.operation.name",
        # --- 3. Request Parameters (Hyperparameters) ---
        "langsmith.metadata.ls_temperature": "gen_ai.request.temperature",
        "langsmith.metadata.ls_max_tokens": "gen_ai.request.max_tokens",
        "langsmith.metadata.ls_top_p": "gen_ai.request.top_p",
        "langsmith.metadata.ls_top_k": "gen_ai.request.top_k",
        "langsmith.metadata.ls_presence_penalty": "gen_ai.request.presence_penalty",
        "langsmith.metadata.ls_frequency_penalty": "gen_ai.request.frequency_penalty",
        "langsmith.metadata.ls_seed": "gen_ai.request.seed",
        "langsmith.metadata.ls_stop_sequences": "gen_ai.request.stop_sequences",
        # ⚠️  CUSTOM: Alternative parameter naming
        "langsmith.temperature": "gen_ai.request.temperature",
        "langsmith.max_tokens": "gen_ai.request.max_tokens",
        "langsmith.top_p": "gen_ai.request.top_p",
        "langsmith.top_k": "gen_ai.request.top_k",
        # --- 4. Usage Metrics (Tokens) ---
        "gen_ai.token.usage.input": "gen_ai.usage.input_tokens",
        "gen_ai.token.usage.output": "gen_ai.usage.output_tokens",
        "gen_ai.token.usage.total": "gen_ai.usage.total_tokens",
        "langsmith.token_usage.prompt_tokens": "gen_ai.usage.input_tokens",
        "langsmith.token_usage.completion_tokens": "gen_ai.usage.output_tokens",
        "langsmith.token_usage.total_tokens": "gen_ai.usage.total_tokens",
        "langsmith.usage.prompt_tokens": "gen_ai.usage.input_tokens",
        "langsmith.usage.completion_tokens": "gen_ai.usage.output_tokens",
        "langsmith.usage.total_tokens": "gen_ai.usage.total_tokens",
        # --- 5. Tool & Function Calling ---
        "langsmith.session_id": "gen_ai.conversation.id",
        "langsmith.thread_id": "gen_ai.conversation.id",
        "langsmith.run_id": "gen_ai.run.id",
        "langsmith.parent_run_id": "gen_ai.parent_run.id",
        # --- 7. Agent & Workflow Attributes ---
        # ⚠️  CUSTOM: Agent-related attributes
        "langsmith.agent.name": "gen_ai.agent.name",
        "langsmith.agent.type": "gen_ai.agent.type",
        "langsmith.agent.description": "gen_ai.agent.description",
        "langsmith.workflow.name": "gen_ai.workflow.name",
        "langsmith.chain.name": "gen_ai.workflow.name",
        # --- 8. Error & Status Handling ---
        "langsmith.error": "gen_ai.error.message",
        "langsmith.error.type": "gen_ai.error.type",
        "langsmith.status": "gen_ai.response.status",
        # --- 9. Response Metadata ---
        "langsmith.response.model": "gen_ai.response.model",
        "langsmith.response.id": "gen_ai.response.id",
        "langsmith.finish_reason": "gen_ai.response.finish_reasons",
        # --- 10. Embeddings ---
        "langsmith.embedding_dimension": "gen_ai.embeddings.dimension.count",
        "langsmith.embedding_model": "gen_ai.request.model",
    }
}

# Default span name transformation mappings
_DEFAULT_NAME_TRANSFORMATIONS = {
    "chat *": "genai.chat",
    "ChatOpenAI*": "genai.chat",
    "ChatAnthropic*": "genai.chat",
    "ChatGoogleGenerativeAI*": "genai.chat",
    "LLMChain*": "genai.chain",
    "AgentExecutor*": "genai.agent",
}

# Global flag to track if processor has been registered (prevents multiple instances)
_PROCESSOR_REGISTERED = False


def enable_langsmith_translator(
    *,
    attribute_transformations: Dict[str, Any] | None = None,
    name_transformations: Dict[str, str] | None = None,
    mutate_original_span: bool = True,
) -> bool:
    """Enable the Langsmith span translator processor.

    This function registers the LangsmithSpanProcessor with the global tracer provider.
    It's safe to call multiple times (idempotent).

    Args:
        attribute_transformations: Custom attribute transformation rules.
            If None, uses default transformations (langsmith.* -> gen_ai.*).
        name_transformations: Custom span name transformation rules.
            If None, uses default transformations.
        mutate_original_span: If True, mutate the original span's attributes.
            If False, only create new synthetic spans.

    Returns:
        True if the processor was registered, False if already registered or disabled.

    Example:
        >>> from opentelemetry.util.genai.langsmith import enable_langsmith_translator
        >>> enable_langsmith_translator()
        True

        >>> # With custom transformations
        >>> enable_langsmith_translator(
        ...     attribute_transformations={
        ...         "rename": {"my.custom.attr": "gen_ai.custom.attr"}
        ...     }
        ... )
    """
    # CRITICAL: Check global flag first to prevent multiple processor instances
    global _PROCESSOR_REGISTERED
    if _PROCESSOR_REGISTERED:
        _LOGGER.debug(
            "LangsmithSpanProcessor already registered (global flag); skipping duplicate"
        )
        return False

    # Import here to avoid circular imports
    from ..processor.langsmith_span_processor import LangsmithSpanProcessor

    provider = trace.get_tracer_provider()

    # Check if provider supports span processors
    if not hasattr(provider, "add_span_processor"):
        _LOGGER.warning(
            "Tracer provider does not support span processors. "
            "LangsmithSpanProcessor cannot be registered. "
            "Make sure you're using the OpenTelemetry SDK TracerProvider."
        )
        return False

    # Check for existing processor to avoid duplicates
    for attr_name in ("_active_span_processors", "_span_processors"):
        existing = getattr(provider, attr_name, [])
        if isinstance(existing, (list, tuple)):
            for proc in existing:
                if isinstance(proc, LangsmithSpanProcessor):
                    _LOGGER.debug(
                        "LangsmithSpanProcessor already registered; skipping duplicate"
                    )
                    return False

    try:
        processor = LangsmithSpanProcessor(
            attribute_transformations=attribute_transformations
            or _DEFAULT_ATTR_TRANSFORMATIONS,
            name_transformations=name_transformations
            or _DEFAULT_NAME_TRANSFORMATIONS,
            mutate_original_span=mutate_original_span,
        )
        provider.add_span_processor(processor)
        _PROCESSOR_REGISTERED = True  # Set global flag to prevent duplicates
        _LOGGER.info(
            "LangsmithSpanProcessor registered automatically "
            "(disable with %s=true)",
            _ENV_DISABLE,
        )
        return True
    except (TypeError, ValueError) as config_err:
        # Fail-fast
        _LOGGER.error(
            "Invalid configuration for LangsmithSpanProcessor: %s",
            config_err,
            exc_info=True,
        )
        raise
    except Exception as exc:
        _LOGGER.warning(
            "Failed to register LangsmithSpanProcessor: %s", exc, exc_info=True
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
            "LangsmithSpanProcessor auto-registration skipped (disabled via %s)",
            _ENV_DISABLE,
        )
        return

    # Try immediate registration first
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        # Real provider exists - register immediately
        enable_langsmith_translator()
    else:
        _LOGGER.debug(
            "TracerProvider not ready yet; deferring LangsmithSpanProcessor registration"
        )
        _install_deferred_registration()


def _install_deferred_registration() -> None:
    """Install a hook to register the processor when TracerProvider becomes available."""
    from ..processor.langsmith_span_processor import LangsmithSpanProcessor

    # Check if another translator has already wrapped set_tracer_provider
    current_func = trace.set_tracer_provider
    if hasattr(current_func, "_traceloop_translator_wrapper"):
        _LOGGER.info(
            "Traceloop translator is already installed; "
            "skipping Langsmith translator to avoid conflicts."
        )
        return

    if hasattr(current_func, "_openlit_translator_wrapper"):
        _LOGGER.info(
            "OpenLit translator is already installed; "
            "skipping Langsmith translator to avoid conflicts."
        )
        return

    # Check if we already wrapped it (prevent double-wrapping on re-import)
    if hasattr(current_func, _WRAPPER_MARKER):
        _LOGGER.debug(
            "Langsmith translator wrapper already installed; skipping"
        )
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
                            if isinstance(proc, LangsmithSpanProcessor):
                                already_registered = True
                                break
                    if already_registered:
                        break

                if not already_registered:
                    # Double-check global flag before registering
                    global _PROCESSOR_REGISTERED
                    if _PROCESSOR_REGISTERED:
                        _LOGGER.debug(
                            "LangsmithSpanProcessor already registered (global flag); skipping deferred registration"
                        )
                        return result

                    processor = LangsmithSpanProcessor(
                        attribute_transformations=_DEFAULT_ATTR_TRANSFORMATIONS,
                        name_transformations=_DEFAULT_NAME_TRANSFORMATIONS,
                        mutate_original_span=True,
                    )
                    tracer_provider.add_span_processor(processor)
                    _PROCESSOR_REGISTERED = True  # Set global flag
                    _LOGGER.info(
                        "LangsmithSpanProcessor registered (deferred) after TracerProvider setup"
                    )
        except Exception as exc:
            _LOGGER.debug(
                "Failed to auto-register LangsmithSpanProcessor: %s", exc
            )

        return result

    # Mark the wrapper so we can detect it later
    wrapped_set_tracer_provider._langsmith_translator_wrapper = True  # type: ignore[attr-defined]
    setattr(wrapped_set_tracer_provider, _WRAPPER_MARKER, True)

    # Install the wrapper
    trace.set_tracer_provider = wrapped_set_tracer_provider


# Auto-enable on import (unless disabled)
_auto_enable()


__all__ = [
    "enable_langsmith_translator",
]
