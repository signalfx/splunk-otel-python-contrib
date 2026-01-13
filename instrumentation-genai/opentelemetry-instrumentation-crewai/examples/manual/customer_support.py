"""CrewAI Customer Support Example (manual OpenTelemetry SDK wiring)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

# Import shared app logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from customer_support_app import (  # noqa: E402
    DEFAULT_INPUTS,
    build_customer_support_crew,
    create_cisco_llm,
)

# Disable CrewAI built-in telemetry
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")

ENABLE_CONSOLE_OUTPUT = (
    os.environ.get("OTEL_CONSOLE_OUTPUT", "false").lower() == "true"
)

# Configure Traces
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
if ENABLE_CONSOLE_OUTPUT:
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# Configure Metrics
metric_readers = [
    PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=60000,
    )
]
if ENABLE_CONSOLE_OUTPUT:
    metric_readers.append(
        PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=60000,
        )
    )
meter_provider = metrics_sdk.MeterProvider(metric_readers=metric_readers)
metrics.set_meter_provider(meter_provider)


CrewAIInstrumentor().instrument(
    tracer_provider=tracer_provider, meter_provider=meter_provider
)


def flush_telemetry() -> None:
    print("\n[FLUSH] Starting telemetry flush", flush=True)
    try:
        tp = trace.get_tracer_provider()
        if hasattr(tp, "force_flush"):
            print("[FLUSH] Flushing traces (timeout=30s)", flush=True)
            tp.force_flush(timeout_millis=30000)
    except Exception as exc:
        print(f"[FLUSH] Warning: Could not flush traces: {exc}", flush=True)

    try:
        mp = metrics.get_meter_provider()
        if hasattr(mp, "force_flush"):
            print("[FLUSH] Flushing metrics (timeout=30s)", flush=True)
            mp.force_flush(timeout_millis=30000)
    except Exception as exc:
        print(f"[FLUSH] Warning: Could not flush metrics: {exc}", flush=True)

    time.sleep(2)
    print("[FLUSH] Telemetry flush complete", flush=True)


if __name__ == "__main__":
    try:
        llm, _token_manager, _ = create_cisco_llm()
        crew, _support_agent, _qa_agent = build_customer_support_crew(llm)

        print("[RUN] Starting CrewAI Customer Support example", flush=True)
        result = crew.kickoff(inputs=DEFAULT_INPUTS)
        print(f"\n[RESULT] {result}", flush=True)
    except Exception as exc:
        print(f"\n[ERROR] Crew execution failed: {exc}", flush=True)
        raise
    finally:
        flush_telemetry()
        print("[DONE] Exiting", flush=True)
        sys.exit(0)

