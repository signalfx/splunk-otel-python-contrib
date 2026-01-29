# pylint: skip-file
"""Manual OpenAI Agents instrumentation example with OAuth2 token support."""

from __future__ import annotations

# ruff: noqa: I001

import os
import sys
import time
import traceback
from typing import Any, cast

from dotenv import load_dotenv
from opentelemetry import _logs, metrics, trace
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
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

from agents import Agent, Runner, function_tool, set_default_openai_client
from openai import AsyncOpenAI

from util import OAuth2TokenManager


# =============================================================================
# LLM Configuration - OAuth2 Provider
# =============================================================================

# Optional app key for request tracking
LLM_APP_KEY = os.environ.get("LLM_APP_KEY")

# Check if we should use OAuth2 or standard OpenAI
USE_OAUTH2 = bool(os.environ.get("LLM_CLIENT_ID"))

# Initialize token manager if OAuth2 credentials are present
token_manager: OAuth2TokenManager | None = None
if USE_OAUTH2:
    token_manager = OAuth2TokenManager()


def get_openai_client() -> AsyncOpenAI:
    """Create OpenAI client with fresh OAuth2 token or standard API key."""
    if USE_OAUTH2 and token_manager:
        token = token_manager.get_token()
        base_url = OAuth2TokenManager.get_llm_base_url(
            os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        )

        # Build extra headers
        extra_headers: dict[str, str] = {"api-key": token}
        if LLM_APP_KEY:
            extra_headers["x-app-key"] = LLM_APP_KEY

        return AsyncOpenAI(
            api_key="placeholder",  # Required but we use api-key header
            base_url=base_url,
            default_headers=extra_headers,
        )
    else:
        # Standard OpenAI API
        return AsyncOpenAI()


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

    # Configure OpenAI client with OAuth2 or standard API key
    client = get_openai_client()
    set_default_openai_client(client)

    if USE_OAUTH2:
        print("[AUTH] Using OAuth2 authentication")
    else:
        print("[AUTH] Using standard OpenAI API key")

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

    print("\n[SUCCESS] Agent execution completed")
    print("Agent response:")
    print(result.final_output)


def flush_telemetry() -> None:
    """Flush all OpenTelemetry providers before exit to ensure traces and metrics are exported."""
    print("\n[FLUSH] Starting telemetry flush", flush=True)

    # Flush traces
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            print("[FLUSH] Flushing traces (timeout=30s)", flush=True)
            tracer_provider.force_flush(timeout_millis=30000)
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush traces: {e}", flush=True)

    # Flush metrics
    try:
        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "force_flush"):
            print("[FLUSH] Flushing metrics (timeout=30s)", flush=True)
            meter_provider.force_flush(timeout_millis=30000)
        if hasattr(meter_provider, "shutdown"):
            print("[FLUSH] Shutting down metrics provider", flush=True)
            meter_provider.shutdown()
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush metrics: {e}", flush=True)

    # Give batch processors time to complete final export
    time.sleep(2)
    print("[FLUSH] Telemetry flush complete\n", flush=True)


def main() -> None:
    load_dotenv()
    _configure_manual_instrumentation()

    exit_code = 0
    try:
        run_agent()
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {e}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        # CRITICAL: Always flush telemetry to ensure spans and metrics are exported
        print("\n" + "=" * 80)
        print("TELEMETRY OUTPUT BELOW")
        print("=" * 80 + "\n")
        flush_telemetry()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
