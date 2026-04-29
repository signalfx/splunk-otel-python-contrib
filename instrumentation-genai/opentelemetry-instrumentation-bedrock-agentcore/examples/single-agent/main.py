# pylint: skip-file
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

"""Bedrock AgentCore Single-Agent Example.

Demonstrates a ReAct-style agentic loop using BedrockAgentCoreApp with the
Amazon Bedrock Converse API (tool calling) and automatic OpenTelemetry
instrumentation via BedrockAgentCoreInstrumentor.

The agent answers weather questions by calling a real weather tool backed by
the Open-Meteo API (no API key required).

Run modes:
1. Default (no CLI args): queries "What is the weather in San Francisco?" and exits.
2. CLI mode: python main.py --city "Paris"

Required environment variables:
    AWS_DEFAULT_REGION   - AWS region (default: us-west-2)
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP collector endpoint (default: http://localhost:4317)

Optional:
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY - explicit AWS credentials
    AWS_PROFILE                               - named credential profile
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT - set to "true" to capture
                                                          message content on spans
"""

import argparse
import os

import boto3
import requests
from dotenv import load_dotenv

# NOTE: OpenTelemetry Python Logs and Events APIs are in beta
from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.bedrock_agentcore import BedrockAgentCoreInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

load_dotenv()

# --- OTEL environment defaults ---
os.environ.setdefault("OTEL_SERVICE_NAME", "bedrock-agentcore-single-agent")

# --- Configure resource ---
resource = Resource(
    attributes={
        ResourceAttributes.SERVICE_NAME: os.environ.get(
            "OTEL_SERVICE_NAME", "bedrock-agentcore-single-agent"
        ),
    }
)

# --- Configure tracing ---
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# --- Configure logging / events ---
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
_logs.set_logger_provider(logger_provider)

# --- Configure metrics ---
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
)
metrics.set_meter_provider(meter_provider)

# --- Instrument Bedrock AgentCore ---
BedrockAgentCoreInstrumentor().instrument(
    tracer_provider=tracer_provider,
    logger_provider=logger_provider,
    meter_provider=meter_provider,
)


# ---------------------------------------------------------------------------
# Tool implementation — uses the free Open-Meteo API (no key required)
# ---------------------------------------------------------------------------


def get_weather(location: str) -> dict:
    """Get current weather for a city using the Open-Meteo geocoding and weather APIs."""
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(
        geocode_url,
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results")
    if not results:
        return {"error": f"Location '{location}' not found"}

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    name = results[0].get("name", location)

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=10,
    )
    weather_resp.raise_for_status()
    current = weather_resp.json().get("current_weather", {})

    return {
        "location": name,
        "temperature_celsius": current.get("temperature"),
        "wind_speed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
    }


# Map tool name → callable for dispatch
_TOOLS = {"get_weather": get_weather}

# Bedrock Converse API tool spec
_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a given city or location.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city or location to get weather for, e.g. 'San Francisco'",
                            }
                        },
                        "required": ["location"],
                    }
                },
            }
        }
    ]
}

_SYSTEM_PROMPT = [
    {
        "text": (
            "You are a helpful weather assistant. "
            "Use the get_weather tool to fetch real weather data when asked about weather. "
            "After presenting the data, add a short commentary about the conditions."
        )
    }
]

_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(query: str) -> str:
    """Run a single-turn agentic query against Bedrock with tool calling."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    client = boto3.client("bedrock-runtime", region_name=region)

    messages = [{"role": "user", "content": [{"text": query}]}]

    # ReAct loop: keep dispatching tool calls until the model produces a text reply
    while True:
        response = client.converse(
            modelId=_MODEL_ID,
            system=_SYSTEM_PROMPT,
            messages=messages,
            toolConfig=_TOOL_CONFIG,
        )

        output_message = response["output"]["message"]
        messages.append(output_message)

        stop_reason = response.get("stopReason")

        if stop_reason == "tool_use":
            tool_results = []
            for block in output_message["content"]:
                if block.get("toolUse"):
                    tool_use = block["toolUse"]
                    tool_fn = _TOOLS.get(tool_use["name"])
                    if tool_fn is None:
                        result = {"error": f"Unknown tool: {tool_use['name']}"}
                    else:
                        try:
                            result = tool_fn(**tool_use["input"])
                        except Exception as exc:  # noqa: BLE001
                            result = {"error": str(exc)}

                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"json": result}],
                            }
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        else:
            # Extract text from the final response
            for block in output_message["content"]:
                if "text" in block:
                    return block["text"]
            return ""


# ---------------------------------------------------------------------------
# BedrockAgentCoreApp entrypoint — creates the Workflow span
# ---------------------------------------------------------------------------

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402

app = BedrockAgentCoreApp()


@app.entrypoint
def agent_handler(event):
    """Handle an invocation event.

    The @app.entrypoint decorator creates a Workflow span. The Converse API
    calls inside run_agent produce child LLM spans (via opentelemetry-instrumentation-bedrock
    if installed). Any AgentCore tool usage (MemoryClient, CodeInterpreter, BrowserClient)
    automatically creates child ToolCall spans via BedrockAgentCoreInstrumentor.
    """
    query = event.get("query", "What is the weather in San Francisco?")
    answer = run_agent(query)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bedrock AgentCore single-agent weather demo with OpenTelemetry instrumentation"
    )
    parser.add_argument(
        "--city",
        default="San Francisco",
        help="City to query weather for (default: San Francisco)",
    )
    args = parser.parse_args()

    query = f"What is the weather in {args.city}?"
    print(f"Query: {query}\n")

    result = agent_handler({"query": query})
    print(result.get("answer", ""))

    tracer_provider.force_flush(timeout_millis=5000)
    meter_provider.force_flush(timeout_millis=5000)


if __name__ == "__main__":
    main()
