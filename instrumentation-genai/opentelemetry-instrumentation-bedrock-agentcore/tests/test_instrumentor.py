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

"""Tests for BedrockAgentCoreInstrumentor."""

import os

from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
from opentelemetry.instrumentation.bedrock_agentcore.utils import is_content_enabled


def test_instrumentor_initialization():
    """Test that the instrumentor can be initialized."""
    instrumentor = BedrockAgentCoreInstrumentor()
    assert instrumentor is not None


def test_instrument_uninstrument(tracer_provider, meter_provider):
    """Test basic instrument/uninstrument round-trip."""
    instrumentor = BedrockAgentCoreInstrumentor()

    # Instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Uninstrument
    instrumentor.uninstrument()

    # Should be able to instrument again
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    instrumentor.uninstrument()


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly specified."""
    instrumentor = BedrockAgentCoreInstrumentor()
    deps = instrumentor.instrumentation_dependencies()
    assert "bedrock-agentcore" in str(deps)


# ---------------------------------------------------------------------------
# is_content_enabled
# ---------------------------------------------------------------------------


def test_is_content_enabled_false_by_default():
    """is_content_enabled returns False when env var is unset."""
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
    assert is_content_enabled() is False


def test_is_content_enabled_true_when_set():
    """is_content_enabled returns True when env var is 'true'."""
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
    try:
        assert is_content_enabled() is True
    finally:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


def test_is_content_enabled_case_insensitive():
    """is_content_enabled treats 'TRUE' and 'True' as enabled."""
    for value in ("TRUE", "True", "TrUe"):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = value
        try:
            assert is_content_enabled() is True, f"Expected True for value={value!r}"
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


def test_is_content_enabled_false_for_non_true_values():
    """is_content_enabled returns False for '1', 'yes', 'on', 'false', empty string."""
    for value in ("1", "yes", "on", "false", "0", ""):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = value
        try:
            assert is_content_enabled() is False, f"Expected False for value={value!r}"
        finally:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
