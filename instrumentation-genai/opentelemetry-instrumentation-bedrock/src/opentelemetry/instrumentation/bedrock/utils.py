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

"""Utility helpers for AWS Bedrock Runtime instrumentation."""

import json
from typing import Any, Optional

_ERROR_MAX_LEN = 256


def safe_json_dumps(value: Any) -> str:
    """Serialize a value without raising."""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return safe_str(value)


def safe_str(value: Any) -> str:
    """Convert a value to string without raising."""
    try:
        return str(value)
    except Exception:
        return repr(value)


def truncate_error(error: Exception) -> str:
    """Return a bounded exception message for span error attributes."""
    message = safe_str(error)
    if len(message) > _ERROR_MAX_LEN:
        return message[:_ERROR_MAX_LEN] + "..."
    return message


def parse_json_body(value: Any) -> Optional[dict[str, Any]]:
    """Parse a JSON body only when doing so will not consume a stream.

    Real ``InvokeModel`` responses usually return a StreamingBody. Reading it
    here would change application behavior, so objects with ``read`` are
    intentionally ignored.
    """
    if value is None or hasattr(value, "read"):
        return None
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, bytearray):
            value = bytes(value).decode("utf-8")
        if isinstance(value, str):
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        if isinstance(value, dict):
            return value
    except Exception:
        return None
    return None


def maybe_parse_json(value: Any) -> Any:
    """Parse a JSON string if possible, otherwise return the original value."""
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value
