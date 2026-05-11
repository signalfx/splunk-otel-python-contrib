#!/usr/bin/env python3
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

"""
Example demonstrating Strands Agents SDK instrumentation with OpenTelemetry.

Covers:
  1. Agent invocation telemetry from Strands Agent.invoke_async.
  2. LLM and tool-call telemetry from Strands lifecycle hooks.
  3. Trace, metric, and log/event export through the OpenTelemetry SDK.

Requirements:
    - strands-agents >= 1.0.0
    - AWS credentials configured for Bedrock
    - OpenTelemetry SDK + OTLP exporter
    - A running OpenTelemetry collector (default: http://127.0.0.1:4317)

Usage:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
    python main.py
"""

import logging
import os
import sys
import urllib.request

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.strands import StrandsInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: "strands-example"})

tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")

tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, headers=()))
)
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint, headers=()),
            export_interval_millis=5000,
        ),
        PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=5000
        ),
    ],
)
metrics.set_meter_provider(meter_provider)

logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint, headers=()))
)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))
_logs.set_logger_provider(logger_provider)
_events.set_event_logger_provider(EventLoggerProvider(logger_provider))

# Instrument before importing Strands so Agent.__init__ and stream wrappers apply.
StrandsInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)


def main() -> None:
    """Run a Strands agent with one HTTP tool."""
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands.tools import tool

        @tool
        def fetch_page(url: str) -> str:
            """Fetch a URL and return a short text response body."""
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "strands-example/1.0"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.read().decode("utf-8", errors="replace")[:2000]

        model = BedrockModel(
            model_id=os.getenv(
                "BEDROCK_MODEL_ID",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            )
        )
        agent = Agent(
            model=model,
            system_prompt=(
                "You are a helpful research assistant. Use fetch_page when "
                "you need to retrieve public web content."
            ),
            tools=[fetch_page],
        )

        prompt = os.getenv(
            "STRANDS_PROMPT",
            "Fetch http://httpbin.org/json and summarize what you find.",
        )
        logger.info("Running Strands agent prompt: %s", prompt)
        result = agent(prompt)
        logger.info("Agent response: %s", result)

    except ImportError as exc:
        logger.error("Import error: %s", exc)
        logger.error("Please install: pip install -e '../../[instruments]'")
        sys.exit(1)
    except Exception as exc:
        logger.error("Error running example: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Example completed successfully.")
    tracer_provider.force_flush(timeout_millis=5000)
    meter_provider.force_flush(timeout_millis=5000)
    logger_provider.force_flush(timeout_millis=5000)


if __name__ == "__main__":
    main()
