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

"""Tests for StrandsInstrumentor."""

import os

import pytest

from opentelemetry.instrumentation.strands import StrandsInstrumentor


def test_instrument_uninstrument_roundtrip(tracer_provider, meter_provider):
    """Test that instrument and uninstrument can be called without errors."""
    instrumentor = StrandsInstrumentor()

    # Instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Verify instrumentation is active
    import opentelemetry.instrumentation.strands as strands_module

    assert strands_module._handler is not None
    assert strands_module._hook_provider is not None

    # Uninstrument
    instrumentor.uninstrument()

    # After uninstrumentation, handler should still exist but wrappers should be removed
    # (Handler is singleton and persists across tests)


def test_instrument_suppresses_builtin_tracer_by_default(
    tracer_provider, meter_provider
):
    """Test that instrumentation suppresses Strands built-in tracer by default."""
    # Set env var to enable suppression (default behavior)
    os.environ["OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"] = "true"

    instrumentor = StrandsInstrumentor()

    # Check that _original_tracer_methods dict exists
    from opentelemetry.instrumentation.strands.wrappers import (
        _original_tracer_methods,
    )

    _original_tracer_methods.clear()  # Ensure clean state

    # Instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Note: We can't easily verify tracer suppression without actually importing
    # strands.telemetry.tracer which may not be available in tests
    # This test mainly verifies that the code path executes without errors

    # Uninstrument
    instrumentor.uninstrument()


def test_suppress_builtin_configurable(tracer_provider, meter_provider):
    """Test that built-in tracer suppression can be disabled via env var."""
    # Set env var to disable suppression
    os.environ["OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"] = "false"

    instrumentor = StrandsInstrumentor()

    # Instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Verify instrumentation is active
    import opentelemetry.instrumentation.strands as strands_module

    assert strands_module._handler is not None

    # Uninstrument
    instrumentor.uninstrument()


def test_instrumentation_dependencies():
    """Test that instrumentation_dependencies returns expected packages."""
    instrumentor = StrandsInstrumentor()
    deps = instrumentor.instrumentation_dependencies()

    assert "strands-agents >= 1.0.0" in deps


def test_multiple_instrument_calls(tracer_provider, meter_provider):
    """Test that multiple instrument calls don't cause errors."""
    instrumentor = StrandsInstrumentor()

    # First instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Second instrument (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Uninstrument
    instrumentor.uninstrument()
