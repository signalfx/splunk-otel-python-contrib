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
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
)
from opentelemetry.util.genai.types import ContentCapturingMode

from .callbacks import CompletionCallback

logger = logging.getLogger(__name__)


def is_experimental_mode() -> bool:  # backward stub (always false)
    return False


_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def get_content_capturing_mode() -> ContentCapturingMode:
    """Return capture mode derived from environment variables."""

    # Preferred configuration: boolean flag + explicit mode
    capture_flag = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    )
    if capture_flag is not None:
        if not _is_truthy(capture_flag):
            return ContentCapturingMode.NO_CONTENT
        raw_mode = os.environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
            "span_and_event",
        )
        normalized = (raw_mode or "").strip().lower().replace("-", "_")
        mapping = {
            "event_only": ContentCapturingMode.EVENT_ONLY,
            "events": ContentCapturingMode.EVENT_ONLY,  # synonym
            "span_only": ContentCapturingMode.SPAN_ONLY,
            "span": ContentCapturingMode.SPAN_ONLY,  # synonym
            "span_and_event": ContentCapturingMode.SPAN_AND_EVENT,
            "both": ContentCapturingMode.SPAN_AND_EVENT,  # synonym
            "none": ContentCapturingMode.NO_CONTENT,
        }
        mode = mapping.get(normalized)
        if mode is not None:
            return mode
        logger.warning(
            "%s is not a valid option for `%s`. Must be one of span_only, event_only, span_and_event, none. Defaulting to `SPAN_AND_EVENT`.",
            raw_mode,
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
        )
        return ContentCapturingMode.SPAN_AND_EVENT

    # Legacy fallback: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES
    legacy_value = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES"
    )
    if legacy_value is not None:
        logger.warning(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES is deprecated and ignored. "
            "Use OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT and "
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE instead."
        )
    return ContentCapturingMode.NO_CONTENT


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
