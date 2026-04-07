"""Test configuration for LlamaIndex instrumentation tests."""

import os

import pytest

from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai import handler as genai_handler

_session_span_exporter = InMemorySpanExporter()
_session_metric_reader = InMemoryMetricReader()


@pytest.fixture(autouse=True)
def environment():
    """Reset env and handler singleton for each test."""
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters


@pytest.fixture(scope="session", autouse=True)
def _instrument_once():
    """Instrument LlamaIndex once for the entire test session."""
    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"
    genai_handler.TelemetryHandler._reset_for_testing()

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(_session_span_exporter))

    meter_provider = MeterProvider(metric_readers=[_session_metric_reader])

    instrumentor = LlamaindexInstrumentor()
    instrumentor._is_instrumented_by_opentelemetry = False
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor


@pytest.fixture
def span_exporter():
    """Provide a cleared span exporter for each test."""
    _session_span_exporter.clear()
    yield _session_span_exporter


@pytest.fixture
def metric_reader():
    yield _session_metric_reader


@pytest.fixture
def instrument(_instrument_once):
    """Marker fixture for tests that need instrumentation."""
    yield _instrument_once
