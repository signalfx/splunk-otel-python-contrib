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
OpenTelemetry Cisco AI Defense Instrumentation

Wrapper-based instrumentation for Cisco AI Defense Python SDK using splunk-otel-util-genai.

This instrumentation captures security inspection events from AI Defense,
adding the critical `gen_ai.security.event_id` span attribute for security
event correlation in Splunk APM.

Usage:
    from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor

    # Instrument AI Defense SDK
    AIDefenseInstrumentor().instrument()

    # Your AI Defense code
    from aidefense.runtime import ChatInspectionClient
    client = ChatInspectionClient(api_key="...")
    result = client.inspect_prompt("user input")
    # Spans are automatically created with gen_ai.security.event_id attribute
"""

from opentelemetry.instrumentation.aidefense.instrumentation import (
    AIDefenseInstrumentor,
)
from opentelemetry.instrumentation.aidefense.version import __version__

__all__ = ["AIDefenseInstrumentor", "__version__"]
