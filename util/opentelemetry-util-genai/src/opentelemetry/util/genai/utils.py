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

import logging
import os

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE,
)
from opentelemetry.util.genai.types import ContentCapturingMode

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
