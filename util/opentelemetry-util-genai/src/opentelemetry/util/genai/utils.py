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
import json
import logging
import os
from typing import Optional

from opentelemetry.util._importlib_metadata import (
    entry_points,  # pyright: ignore[reportUnknownVariableType]
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.types import ContentCapturingMode

from .callbacks import CompletionCallback

logger = logging.getLogger(__name__)

_LEGACY_CAPTURE_MESSAGE_CONTENT_MODE = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE"
)


def is_experimental_mode() -> bool:  # backward stub (always false)
    return False


_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def _parse_legacy_mode_fragment(raw_mode: str) -> ContentCapturingMode:
    """Parse deprecated MODE env or invalid-legacy fallback (defaults SPAN_AND_EVENT)."""
    normalized = raw_mode.lower().replace("-", "_")
    mapping = {
        "event_only": ContentCapturingMode.EVENT_ONLY,
        "events": ContentCapturingMode.EVENT_ONLY,
        "span_only": ContentCapturingMode.SPAN_ONLY,
        "span": ContentCapturingMode.SPAN_ONLY,
        "span_and_event": ContentCapturingMode.SPAN_AND_EVENT,
        "both": ContentCapturingMode.SPAN_AND_EVENT,
        "none": ContentCapturingMode.NO_CONTENT,
        "no_content": ContentCapturingMode.NO_CONTENT,
    }
    mode = mapping.get(normalized)
    if mode is not None:
        return mode
    logger.warning(
        "%s is not a valid option for deprecated `%s`. Must be one of "
        "span_only, event_only, span_and_event, none. Defaulting to `SPAN_AND_EVENT`.",
        raw_mode,
        _LEGACY_CAPTURE_MESSAGE_CONTENT_MODE,
    )
    return ContentCapturingMode.SPAN_AND_EVENT


def get_content_capturing_mode() -> ContentCapturingMode:
    """Return capture mode from ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``.

    Aligns with upstream OpenTelemetry GenAI util: the primary variable holds the
    mode. SDOT does not gate on experimental stability mode.

    Legacy: truthy ``true``/``1``/``yes``/``on`` with optional deprecated
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE`` (defaults to
    ``SPAN_AND_EVENT`` when the legacy mode var is unset).
    """
    legacy_messages = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES"
    )
    if legacy_messages is not None:
        logger.warning(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES is deprecated and ignored. "
            "Use OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT instead."
        )

    envvar = os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT)
    if envvar is None or not str(envvar).strip():
        return ContentCapturingMode.NO_CONTENT

    raw = str(envvar).strip()
    upper = raw.upper().replace("-", "_")

    try:
        return ContentCapturingMode[upper]
    except KeyError:
        pass

    normalized = raw.lower().replace("-", "_")
    synonym_to_mode = {
        "span_only": ContentCapturingMode.SPAN_ONLY,
        "span": ContentCapturingMode.SPAN_ONLY,
        "event_only": ContentCapturingMode.EVENT_ONLY,
        "events": ContentCapturingMode.EVENT_ONLY,
        "span_and_event": ContentCapturingMode.SPAN_AND_EVENT,
        "span_and_events": ContentCapturingMode.SPAN_AND_EVENT,
        "both": ContentCapturingMode.SPAN_AND_EVENT,
        "all": ContentCapturingMode.SPAN_AND_EVENT,
        "none": ContentCapturingMode.NO_CONTENT,
        "no_content": ContentCapturingMode.NO_CONTENT,
    }
    if normalized in synonym_to_mode:
        return synonym_to_mode[normalized]

    if _is_truthy(raw):
        legacy_mode = os.environ.get(_LEGACY_CAPTURE_MESSAGE_CONTENT_MODE)
        if legacy_mode is not None and str(legacy_mode).strip():
            logger.warning(
                "%s is deprecated; set the full mode on %s (e.g. SPAN_ONLY, SPAN_AND_EVENT).",
                _LEGACY_CAPTURE_MESSAGE_CONTENT_MODE,
                OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
            )
            return _parse_legacy_mode_fragment(str(legacy_mode).strip())
        return ContentCapturingMode.SPAN_AND_EVENT

    if raw.lower() in ("false", "0", "no", "off"):
        return ContentCapturingMode.NO_CONTENT

    logger.warning(
        "%s is not a valid option for `%s` environment variable. Must be one of %s. Defaulting to `NO_CONTENT`.",
        raw,
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
        ", ".join(e.name for e in ContentCapturingMode),
    )
    return ContentCapturingMode.NO_CONTENT


def should_emit_event() -> bool:
    """Return whether GenAI content events should be emitted.

    If ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` is set to a non-empty value, it
    takes precedence. Otherwise the default follows capture mode (same defaults
    as upstream, without experimental gating).
    """
    envvar = os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT)
    if envvar and envvar.strip():
        envvar_lower = envvar.lower().strip()
        if envvar_lower == "true":
            return True
        if envvar_lower == "false":
            return False
        logger.warning(
            "%s is not a valid option for `%s` environment variable. Must be one of true or false (case-insensitive). Defaulting based on content capturing mode.",
            envvar,
            OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
        )

    content_mode = get_content_capturing_mode()
    if content_mode in (
        ContentCapturingMode.EVENT_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return True
    return False


def should_capture_content_on_spans() -> bool:
    """Return True when span attributes should include message content conversion."""
    mode = get_content_capturing_mode()
    if mode == ContentCapturingMode.NO_CONTENT:
        return False
    if mode == ContentCapturingMode.EVENT_ONLY and not should_emit_event():
        return False
    return True


def _coerce_completion_callback(
    provider: object, name: str
) -> CompletionCallback | None:
    if provider is None:
        return None
    if hasattr(provider, "on_completion"):
        if isinstance(provider, type):
            try:
                instance = provider()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Completion callback class '%s' failed to instantiate: %s",
                    name,
                    exc,
                )
                return None
            return instance  # type: ignore[return-value]
        return provider  # type: ignore[return-value]
    if callable(provider):
        try:
            instance = provider()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Completion callback factory '%s' raised an exception: %s",
                name,
                exc,
            )
            return None
        if hasattr(instance, "on_completion"):
            return instance  # type: ignore[return-value]
        logger.warning(
            "Completion callback factory '%s' returned an object without on_completion",
            name,
        )
        return None
    logger.warning(
        "Completion callback entry point '%s' is not callable or instance",
        name,
    )
    return None


def load_completion_callbacks(
    selected: set[str] | None,
) -> tuple[list[tuple[str, CompletionCallback]], set[str]]:
    callbacks: list[tuple[str, CompletionCallback]] = []
    seen: set[str] = set()
    try:
        entries = entry_points(
            group="opentelemetry_util_genai_completion_callbacks"
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("Completion callback entry point group not available")
        return callbacks, seen
    for ep in entries:  # type: ignore[assignment]
        name = getattr(ep, "name", "")
        lowered = name.lower()
        seen.add(lowered)
        if selected and lowered not in selected:
            continue
        try:
            provider = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to load completion callback '%s': %s",
                name,
                exc,
                exc_info=True,
            )
            continue
        instance = _coerce_completion_callback(provider, name)
        if instance is None:
            continue
        callbacks.append((name, instance))
    return callbacks, seen


def is_truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def should_capture_tool_definitions() -> bool:
    """Check whether tool definition capture is opted-in.

    Returns ``True`` only when
    :envvar:`OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS` is set to a
    truthy value.  Instrumentations should call this **before** serializing
    tool schemas to avoid the cost of JSON-encoding potentially large payloads
    when the feature is disabled.
    """
    return is_truthy_env(
        os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS)
    )


def parse_callback_filter(value: Optional[str]) -> set[str] | None:
    if value is None:
        return None
    selected = {
        item.strip().lower() for item in value.split(",") if item.strip()
    }
    return selected or None


def gen_ai_json_dumps(value: object) -> str:
    """
    Serialize GenAI payloads to JSON.

    This is the helper expected by openai-agents-v2 span_processor.safe_json_dumps.
    It should behave like json.dumps, but can be extended later (e.g., custom encoder).
    """
    return json.dumps(value, ensure_ascii=False)
