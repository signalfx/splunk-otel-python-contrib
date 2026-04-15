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
Example demonstrating Strands + Bedrock AgentCore instrumentation with OpenTelemetry.

Covers all three HYBIM-448 requirements:
  1. Agents and tools telemetry (Strands Agents SDK)
  2. AgentCore components monitoring (Memory, CodeInterpreter, Browser)
  3. Observability integration - telemetry forwarded to collector / Splunk Platform

All components run within a single BedrockAgentCoreApp entrypoint, producing a
cohesive span hierarchy:

  Workflow
  ├── RetrievalInvocation  (memory.retrieve_memories - load prior context)
  ├── AgentInvocation
  │    ├── LLMInvocation
  │    └── ToolCall        (fetch_page)
  ├── ToolCall             (memory.create_event - store result)
  ├── ToolCall             (code_interpreter.start / execute_code / stop)
  └── ToolCall             (browser.start / get_session / stop)

Requirements:
    - strands-agents >= 1.0.0
    - bedrock-agentcore >= 1.0.0
    - AWS credentials configured for Bedrock and AgentCore
    - OpenTelemetry SDK + OTLP exporter
    - A running OpenTelemetry collector (default: http://127.0.0.1:4317)

Usage:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
    export MEMORY_ID=<your-agentcore-memory-id>   # optional
    python main.py
"""

import logging
import os
import sys
import urllib.request

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.strands import StrandsInstrumentor
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

# ---------------------------------------------------------------------------
# Requirement 3: Observability integration
# Set up trace + metric providers exporting to collector and console
# ---------------------------------------------------------------------------
resource = Resource(
    attributes={ResourceAttributes.SERVICE_NAME: "strands-agentcore-example"}
)

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

# Instrument before any imports from Strands or AgentCore
StrandsInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)


def main():
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    memory_id = os.getenv("MEMORY_ID")

    try:
        from bedrock_agentcore import BedrockAgentCoreApp
        from bedrock_agentcore.memory.client import MemoryClient
        from bedrock_agentcore.tools.browser_client import BrowserClient
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands.tools import tool

        # ------------------------------------------------------------------
        # Requirement 1: Agent tool definitions
        # ------------------------------------------------------------------
        @tool
        def fetch_page(url: str) -> str:
            """Fetch the contents of a URL and return the response body as text."""
            req = urllib.request.Request(
                url, headers={"User-Agent": "strands-example/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8", errors="replace")[:2000]

        model = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")

        agent = Agent(
            model=model,
            system_prompt="You are a helpful research assistant. Use the fetch_page tool to look up real information when needed.",
            tools=[fetch_page],
        )

        # ------------------------------------------------------------------
        # Requirement 2: AgentCore component clients
        # ------------------------------------------------------------------
        memory_client = MemoryClient(region_name=region) if memory_id else None
        interpreter = CodeInterpreter(region=region)
        browser = BrowserClient(region=region)

        # ------------------------------------------------------------------
        # Single entrypoint wrapping all components under one Workflow span
        # ------------------------------------------------------------------
        app = BedrockAgentCoreApp()

        @app.entrypoint
        def run(payload):
            query = payload.get(
                "query", "Fetch http://httpbin.org/json and summarize it."
            )

            # 1. Memory: load prior context (RetrievalInvocation span)
            if memory_client:
                try:
                    prior = memory_client.retrieve_memories(
                        memory_id=memory_id,
                        namespace="example",
                        query=query,
                    )
                    logger.info(f"Memory retrieved {len(prior)} prior record(s)")
                except Exception as e:
                    logger.warning(f"Memory retrieve skipped: {e}")

            # 2. Agent: research the query (AgentInvocation → LLMInvocation + ToolCall spans)
            agent_result = agent(query)
            logger.info(f"Agent response: {agent_result}")

            # 3. Memory: store the result (ToolCall span)
            if memory_client:
                try:
                    memory_client.create_event(
                        memory_id=memory_id,
                        actor_id="example-user",
                        session_id="example-session",
                        messages=[("user", query), ("assistant", str(agent_result))],
                    )
                    logger.info("Memory event stored")
                except Exception as e:
                    logger.warning(f"Memory store skipped: {e}")

            # 4. CodeInterpreter: run a quick computation (ToolCall spans)
            try:
                interpreter.start()
                interpreter.execute_code("print('AgentCore CodeInterpreter active')")
                interpreter.stop()
                logger.info("CodeInterpreter demo completed")
            except Exception as e:
                logger.warning(
                    f"CodeInterpreter skipped (requires AgentCore access): {e}"
                )

            # 5. Browser: demonstrate session lifecycle (ToolCall spans)
            try:
                browser.start()
                browser.get_session()
                browser.stop()
                logger.info("Browser demo completed")
            except Exception as e:
                logger.warning(f"Browser skipped (requires AgentCore access): {e}")

            return agent_result

        logger.info("Running combined AgentCore workflow...")
        run(
            {
                "query": "Fetch the content of http://httpbin.org/json and summarize what you find."
            }
        )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install: pip install -e '../../[instruments]'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Example completed successfully!")

    tracer_provider.force_flush(timeout_millis=5000)
    meter_provider.force_flush(timeout_millis=5000)


if __name__ == "__main__":
    main()
