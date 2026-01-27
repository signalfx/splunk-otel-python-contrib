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

"""Test configuration for FastMCP instrumentation tests."""

import pytest
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


@pytest.fixture(scope="session")
def span_exporter():
    """Create an in-memory span exporter for testing."""
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="session")
def tracer_provider(span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture(scope="session")
def metric_reader():
    """Create an in-memory metric reader for testing."""
    return InMemoryMetricReader()


@pytest.fixture(scope="session")
def meter_provider(metric_reader):
    """Create a meter provider with in-memory reader."""
    return MeterProvider(metric_readers=[metric_reader])


@pytest.fixture
def mock_telemetry_handler():
    """Create a mock TelemetryHandler for testing."""
    handler = MagicMock()
    handler.start_tool_call = MagicMock()
    handler.stop_tool_call = MagicMock()
    handler.fail_tool_call = MagicMock()
    handler.start_agent = MagicMock()
    handler.stop_agent = MagicMock()
    handler.fail_agent = MagicMock()
    handler.start_step = MagicMock()
    handler.stop_step = MagicMock()
    handler.fail_step = MagicMock()
    return handler


@pytest.fixture
def clear_spans(span_exporter):
    """Clear spans before and after each test."""
    span_exporter.clear()
    yield
    span_exporter.clear()
