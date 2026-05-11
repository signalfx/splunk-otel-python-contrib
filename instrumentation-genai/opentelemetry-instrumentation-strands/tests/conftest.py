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

"""Pytest configuration and fixtures for Strands instrumentation tests."""

import os

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


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
    import opentelemetry.instrumentation.strands as strands_module
    from opentelemetry.util.genai.handler import get_telemetry_handler

    # Store original value
    original_handler = strands_module._handler

    # Reset before test
    strands_module._handler = None
    if hasattr(get_telemetry_handler, "_default_handler"):
        delattr(get_telemetry_handler, "_default_handler")

    yield

    # Reset after test
    strands_module._handler = original_handler
    if hasattr(get_telemetry_handler, "_default_handler"):
        delattr(get_telemetry_handler, "_default_handler")


@pytest.fixture(autouse=True)
def environment():
    """Set up test environment variables."""
    original_evals = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS")
    original_emitters = os.environ.get("OTEL_INSTRUMENTATION_GENAI_EMITTERS")
    original_suppress = os.environ.get(
        "OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"
    )

    os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"
    os.environ["OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"] = "false"

    yield

    if original_evals is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = original_evals

    if original_emitters is None:
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_EMITTERS", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = original_emitters

    if original_suppress is None:
        os.environ.pop("OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER", None)
    else:
        os.environ["OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER"] = (
            original_suppress
        )


class StubTelemetryHandler:
    """A stub telemetry handler for testing wrapper functions."""

    def __init__(self):
        self.started_workflows = []
        self.stopped_workflows = []
        self.started_agents = []
        self.stopped_agents = []
        self.started_llm = []
        self.stopped_llm = []
        self.started_tool_calls = []
        self.stopped_tool_calls = []
        self.started_retrievals = []
        self.stopped_retrievals = []
        self.failed_entities = []

    def start_workflow(self, workflow):
        self.started_workflows.append(workflow)
        return workflow

    def stop_workflow(self, workflow):
        self.stopped_workflows.append(workflow)
        return workflow

    def fail_workflow(self, workflow, error):
        self.failed_entities.append((workflow, error))
        return workflow

    def start_agent(self, agent):
        self.started_agents.append(agent)
        return agent

    def stop_agent(self, agent):
        self.stopped_agents.append(agent)
        return agent

    def fail_agent(self, agent, error):
        self.failed_entities.append((agent, error))
        return agent

    def start_llm(self, invocation):
        self.started_llm.append(invocation)
        return invocation

    def stop_llm(self, invocation):
        self.stopped_llm.append(invocation)
        return invocation

    def fail_llm(self, invocation, error):
        self.failed_entities.append((invocation, error))
        return invocation

    def start_tool_call(self, tool_call):
        self.started_tool_calls.append(tool_call)
        return tool_call

    def stop_tool_call(self, tool_call):
        self.stopped_tool_calls.append(tool_call)
        return tool_call

    def fail_tool_call(self, tool_call, error):
        self.failed_entities.append((tool_call, error))
        return tool_call

    def start_retrieval(self, invocation):
        self.started_retrievals.append(invocation)
        return invocation

    def stop_retrieval(self, invocation):
        self.stopped_retrievals.append(invocation)
        return invocation

    def fail_retrieval(self, invocation, error):
        self.failed_entities.append((invocation, error))
        return invocation


@pytest.fixture
def stub_handler():
    """Create a stub telemetry handler for testing."""
    return StubTelemetryHandler()
