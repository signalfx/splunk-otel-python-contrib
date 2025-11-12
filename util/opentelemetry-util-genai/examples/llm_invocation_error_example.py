#!/usr/bin/env python3
"""
Example: Evaluation Error Example

Install deepeval for evaluation:
    pip install deepeval

This example demonstrates:
1. Successful LLM call, with a failed evaluation
"""

import os

os.environ["OTEL_RESOURCE_ATTRIBUTES"] = (
    "deployment.environment=example_metric_errors"
)
os.environ["OTEL_SERVICE_NAME"] = "demo-app-util-genai-dev"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE"] = (
    "SPAN_AND_EVENT"
)
os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"

import time

from opentelemetry import _logs as logs
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    ConsoleLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)


def setup_telemetry():
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(trace_provider)

    metric_reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=5000
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(ConsoleLogExporter())
    )
    logs.set_logger_provider(logger_provider)

    return trace_provider, meter_provider, logger_provider


def successful_llm_call():
    handler = get_telemetry_handler()
    print("Starting successful LLM invocation...")
    llm = LLMInvocation(
        request_model="gpt-5-nano",
        input_messages=[
            InputMessage(
                role="user",
                parts=[Text(content="Hello, how can I track my order?")],
            ),
        ],
    )
    handler.start_llm(llm)
    time.sleep(0.1)
    llm.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[
                Text(
                    content="You can track your order using the tracking link sent to your email."
                )
            ],
            finish_reason="stop",
        )
    ]
    llm.input_tokens = 12
    llm.output_tokens = 18
    handler.stop_llm(llm)
    print("LLM invocation completed successfully.\n")


def failed_llm_call():
    handler = get_telemetry_handler()
    print("Starting failed LLM invocation...")
    llm = LLMInvocation(
        request_model="gpt-5-nano",
        input_messages=[
            InputMessage(
                role="user",
                parts=[Text(content="Tell me about the weather in Atlantis.")],
            ),
        ],
    )
    handler.start_llm(llm)
    time.sleep(0.1)
    error = Error(message="Model unavailable", type=RuntimeError)
    handler.fail_llm(llm, error)
    print("LLM invocation failed.\n")


if __name__ == "__main__":
    setup_telemetry()
    successful_llm_call()
    failed_llm_call()
    time.sleep(6)  # Wait for metrics export
