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

"""Pytest configuration and fixtures for AI Defense instrumentation tests."""

import os

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


@pytest.fixture(scope="function")
def span_exporter():
    """Create an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function")
def metric_reader():
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function")
def tracer_provider(span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def meter_provider(metric_reader):
    """Create a meter provider with in-memory reader."""
    provider = MeterProvider(metric_readers=[metric_reader])
    return provider


@pytest.fixture(autouse=True)
def reset_global_handler():
    """Reset the global handler before and after each test."""
    import opentelemetry.instrumentation.aidefense.instrumentation as aidefense_module
    from opentelemetry.util.genai.handler import TelemetryHandler

    original_handler = aidefense_module._handler

    aidefense_module._handler = None
    TelemetryHandler._reset_for_testing()

    yield

    aidefense_module._handler = original_handler
    TelemetryHandler._reset_for_testing()


@pytest.fixture(autouse=True)
def environment():
    """Set up test environment variables."""
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters
