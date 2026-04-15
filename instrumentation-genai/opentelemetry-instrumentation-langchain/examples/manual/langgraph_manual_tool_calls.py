#!/usr/bin/env python3
"""LangGraph + manual handler instrumentation example.

Demonstrates combining auto-instrumented LangGraph agent spans with manual
handler.start_tool_call() / handler.start_step() calls inside async node
bodies.  All spans end up on the same trace with correct parent-child
relationships.

Usage:
    # Start an OTel collector on localhost:4317, then:
    export OTEL_SERVICE_NAME=langgraph-manual-demo
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
    python langgraph_manual_tool_calls.py
"""

import asyncio

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_instrumentation():
    """Set up OTel providers and SDOT instrumentors."""
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )
    _events.set_event_logger_provider(EventLoggerProvider())

    from opentelemetry.instrumentation.langchain import LangchainInstrumentor

    LangchainInstrumentor().instrument()


# Must be called before importing LangGraph / LangChain
configure_instrumentation()

from typing import TypedDict  # noqa: E402

from langgraph.graph import StateGraph  # noqa: E402
from opentelemetry.util.genai.handler import get_telemetry_handler  # noqa: E402
from opentelemetry.util.genai.types import ToolCall, Step  # noqa: E402


class State(TypedDict):
    query: str
    result: str


# -- Async node functions with manual tool/step instrumentation --


async def lookup_node(state: State) -> State:
    """Async node: looks up data using a manual ToolCall span."""
    handler = get_telemetry_handler()

    # Create a manual ToolCall — the handler automatically parents it
    # to the current agent span (even inside LangGraph async nodes).
    tool = ToolCall(
        name="database_lookup",
        tool_type="datastore",
        tool_description="Look up customer record in database",
        arguments={"query": state["query"]},
        agent_name="lookup-agent",
    )
    handler.start_tool_call(tool)

    # Simulate database query
    await asyncio.sleep(0.05)
    result = f"Found record for: {state['query']}"

    tool.tool_result = result
    handler.stop_tool_call(tool)

    return {"query": state["query"], "result": result}


async def analysis_node(state: State) -> State:
    """Async node: analyses data using a manual Step span."""
    handler = get_telemetry_handler()

    step = Step(
        name="sentiment_analysis",
        objective="Analyse sentiment of the query",
        step_type="processing",
        source="agent",
        assigned_agent="analysis-agent",
    )
    handler.start_step(step)

    await asyncio.sleep(0.03)
    analysis = f"Analysis complete for: {state['result']}"

    handler.stop_step(step)

    return {"query": state["query"], "result": analysis}


async def main():
    builder = StateGraph(State)

    # metadata={"agent_name": ...} makes the auto-instrumented span carry
    # gen_ai.agent.name, and enables the handler's agent context stack.
    builder.add_node("lookup", lookup_node, metadata={"agent_name": "lookup-agent"})
    builder.add_node(
        "analysis", analysis_node, metadata={"agent_name": "analysis-agent"}
    )
    builder.add_edge("__start__", "lookup")
    builder.add_edge("lookup", "analysis")
    builder.add_edge("analysis", "__end__")

    graph = builder.compile()

    result = await graph.ainvoke({"query": "customer 42 order status", "result": ""})
    print(f"Result: {result['result']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Flush all providers before exit
        try:
            trace.get_tracer_provider().force_flush(timeout_millis=5000)
        except Exception:
            pass
        try:
            metrics.get_meter_provider().force_flush(timeout_millis=5000)
        except Exception:
            pass
        try:
            _logs.get_logger_provider().force_flush(timeout_millis=5000)
        except Exception:
            pass
