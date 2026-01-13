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

"""Utility modules for AI Defense instrumentation."""

from opentelemetry.instrumentation.aidefense.util.helper import (
    add_event_id_to_current_span,
    create_ai_defense_invocation,
    create_input_message,
    execute_with_telemetry,
    get_server_address,
    AI_DEFENSE_MODEL,
    AI_DEFENSE_SYSTEM,
    AI_DEFENSE_FRAMEWORK,
    AI_DEFENSE_OPERATION,
    MAX_CONTENT_LENGTH,
    MAX_SHORT_CONTENT_LENGTH,
    MAX_MESSAGES_IN_CONVERSATION,
)

__all__ = [
    "add_event_id_to_current_span",
    "create_ai_defense_invocation",
    "create_input_message",
    "execute_with_telemetry",
    "get_server_address",
    "AI_DEFENSE_MODEL",
    "AI_DEFENSE_SYSTEM",
    "AI_DEFENSE_FRAMEWORK",
    "AI_DEFENSE_OPERATION",
    "MAX_CONTENT_LENGTH",
    "MAX_SHORT_CONTENT_LENGTH",
    "MAX_MESSAGES_IN_CONVERSATION",
]

