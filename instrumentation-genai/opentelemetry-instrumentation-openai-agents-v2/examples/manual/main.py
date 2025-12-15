# pylint: skip-file
"""Manual OpenAI Agents instrumentation example."""

from __future__ import annotations

# ruff: noqa: I001

from typing import Any, cast

from dotenv import load_dotenv
from opentelemetry import _logs
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.openai_agents.span_processor import (  # noqa: E402
    stop_workflow,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

from agents import Agent, Runner, function_tool


def _configure_manual_instrumentation() -> None:
    """Configure tracing/metrics/logging manually so exported data goes to OTLP."""

    # Traces
    set_tracer_provider(TracerProvider())
    tracer_provider = cast(Any, get_tracer_provider())
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    # Metrics
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    # Logs
    _logs.set_logger_provider(LoggerProvider())
    logger_provider = cast(Any, _logs.get_logger_provider())
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )

    # OpenAI Agents instrumentation
    instrumentor: Any = OpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=get_tracer_provider())


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather response for the requested city."""

    return f"The forecast for {city} is sunny with pleasant temperatures."


def run_agent() -> None:
    """Create a simple agent and execute a single run."""

    assistant = Agent(
        name="Travel Concierge",
        instructions=(
            "You are a concise travel concierge. Use the weather tool when the"
            " traveler asks about local conditions."
        ),
        tools=[get_weather],
    )

    result = Runner.run_sync(
        assistant,
        "I'm visiting Barcelona this weekend. How should I pack?",
    )

    print("Agent response:")
    print(result.final_output)


@function_tool
def get_budget(city: str) -> str:
    """Return a simplified travel budget response for the requested city."""

    return f"The plan for {city} is budget-friendly."


def run_agent1() -> None:
    """Create a simple agent and execute a single run."""

    assistant = Agent(
        name="Travel Budget",
        instructions=(
            "You are a concise travel budget planner. Use the budget tool when the"
            " traveler asks about local conditions."
        ),
        tools=[get_budget],
    )

    result = Runner.run_sync(
        assistant,
        "I'm visiting Barcelona this weekend. How to plan my budget?",
    )

    print("Agent response:")
    print(result.final_output)


def main() -> None:
    load_dotenv()
    _configure_manual_instrumentation()
    try:
        run_agent()
        # run_agent1()
    finally:
        # Stop workflow to finalize the workflow span and flush traces
        stop_workflow()
        # Force flush to ensure all spans are exported
        tracer_provider = cast(Any, get_tracer_provider())
        tracer_provider.force_flush()


if __name__ == "__main__":
    main()
