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

"""Utility functions for Bedrock AgentCore instrumentation."""

import json
from os import environ
from typing import Any

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def is_content_enabled() -> bool:
    """Check if content capture is enabled via environment variable.

    Returns:
        True if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is set to 'true',
        False otherwise.
    """
    return (
        environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false").lower()
        == "true"
    )


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize an object to JSON, with fallback on error.

    Args:
        obj: Object to serialize

    Returns:
        JSON string, or repr(obj) if JSON serialization fails
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return repr(obj)


def safe_str(value: Any) -> str:
    """Safely convert any value to string, never raising exceptions.

    Args:
        value: Any value to convert

    Returns:
        String representation of the value
    """
    try:
        return str(value)
    except Exception:
        return repr(value)
