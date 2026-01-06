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

"""Tests for AI Defense instrumentation."""

import pytest
from unittest.mock import MagicMock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.aidefense import AIDefenseInstrumentor
from opentelemetry.instrumentation.aidefense.instrumentation import (
    _populate_invocation_from_result,
)
from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.genai.attributes import GEN_AI_SECURITY_EVENT_ID


@pytest.fixture
def tracer_provider():
    """Create a tracer provider with in-memory exporter for testing."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(
        trace.get_tracer_provider().get_tracer(__name__)._span_processor
        if hasattr(trace.get_tracer_provider().get_tracer(__name__), "_span_processor")
        else None
    )
    trace.set_tracer_provider(provider)
    return provider, exporter


class TestAIDefenseInstrumentor:
    """Test AIDefenseInstrumentor class."""

    def test_instrumentation_dependencies(self):
        """Test that instrumentation dependencies are correctly specified."""
        instrumentor = AIDefenseInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "cisco-aidefense-sdk >= 2.0.0" in deps

    def test_instrument_uninstrument(self):
        """Test that instrument and uninstrument don't raise errors."""
        instrumentor = AIDefenseInstrumentor()

        # Should not raise even if ai-defense-python-sdk is not installed
        try:
            instrumentor.instrument()
        except ModuleNotFoundError:
            # Expected if ai-defense-python-sdk is not installed
            pass

        try:
            instrumentor.uninstrument()
        except Exception:
            # Should not fail even if not instrumented
            pass


class TestPopulateInvocationFromResult:
    """Test _populate_invocation_from_result helper function."""

    def test_populate_with_event_id(self):
        """Test that event_id is captured as security_event_id field."""
        # Create mock result
        result = MagicMock()
        result.event_id = "test-event-id-123"
        result.is_safe = True
        result.action = MagicMock(value="Allow")

        # Create invocation
        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            operation="chat",
        )

        # Populate
        _populate_invocation_from_result(invocation, result)

        # Verify event_id is captured
        assert invocation.security_event_id == "test-event-id-123"
        # Verify output_messages contains action and is_safe
        assert len(invocation.output_messages) == 1
        assert "action=Allow" in invocation.output_messages[0].parts[0].content
        assert "is_safe=True" in invocation.output_messages[0].parts[0].content

    def test_populate_with_blocked_result(self):
        """Test that blocked results are properly captured."""
        # Create mock result with block action
        result = MagicMock()
        result.event_id = "violation-event-456"
        result.is_safe = False
        result.action = MagicMock(value="Block")

        # Create invocation
        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            operation="chat",
        )

        # Populate
        _populate_invocation_from_result(invocation, result)

        # Verify event_id is captured
        assert invocation.security_event_id == "violation-event-456"
        # Verify output shows blocked
        assert len(invocation.output_messages) == 1
        assert "action=Block" in invocation.output_messages[0].parts[0].content
        assert "is_safe=False" in invocation.output_messages[0].parts[0].content

    def test_populate_without_event_id(self):
        """Test handling when event_id is None."""
        result = MagicMock()
        result.event_id = None
        result.is_safe = True
        result.action = MagicMock(value="Allow")

        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            operation="chat",
        )

        _populate_invocation_from_result(invocation, result)

        # event_id should be None
        assert invocation.security_event_id is None


class TestSecurityEventIdAttribute:
    """Test that security_event_id is correctly set and emitted."""

    def test_security_event_id_constant(self):
        """Verify the semconv constant has expected value."""
        assert GEN_AI_SECURITY_EVENT_ID == "gen_ai.security.event_id"

    def test_invocation_security_event_id_field(self):
        """Test that LLMInvocation has security_event_id field."""
        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            operation="chat",
            security_event_id="test-123",
        )

        assert invocation.security_event_id == "test-123"

        # Verify semantic convention attributes include security_event_id
        semconv_attrs = invocation.semantic_convention_attributes()
        assert semconv_attrs.get(GEN_AI_SECURITY_EVENT_ID) == "test-123"

    def test_invocation_without_security_event_id(self):
        """Test that security_event_id is not emitted when None."""
        invocation = LLMInvocation(
            request_model="cisco-ai-defense",
            operation="chat",
        )

        assert invocation.security_event_id is None

        # Verify it's not in semantic convention attributes
        semconv_attrs = invocation.semantic_convention_attributes()
        assert GEN_AI_SECURITY_EVENT_ID not in semconv_attrs
