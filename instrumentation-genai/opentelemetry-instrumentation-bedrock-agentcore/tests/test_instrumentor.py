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

from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor


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
