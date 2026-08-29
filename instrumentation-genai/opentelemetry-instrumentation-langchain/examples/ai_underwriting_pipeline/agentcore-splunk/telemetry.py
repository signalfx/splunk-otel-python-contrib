"""OpenTelemetry wiring for the AI underwriting pipeline.

Two export modes, selected by the ``UNDERWRITING_EXPORT_MODE`` environment variable,
because the two of them demonstrate opposite halves of the same argument:

``otlp`` (default)
    Export spans over OTLP to a collector or straight to Agent Observability. This is
    the *properly instrumented* case -- telemetry leaves the process and arrives in AO,
    so AO can evaluate it and the workload shows up in AO's own inventory.

``console``
    Export spans as OTLP-shaped JSON to **stdout**. Under AgentCore, stdout is captured
    into CloudWatch Logs, so the spans are really produced but never reach a collector.
    This reproduces a pattern seen in the wild -- an application that genuinely runs
    OpenTelemetry and then exports it to a log sink -- and it is the input for the
    "recover spans from logs and forward them to AO" path. Crucially these are the
    *application's own* spans, so unlike AWS platform spans they carry ``gen_ai.*``
    semantic-convention attributes, which is what makes recovery worth doing.

``both``
    Both of the above, for side-by-side comparison in one run.

Nothing here invents a span. The console mode changes only the transport.
"""

from __future__ import annotations

import os
import sys

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

EXPORT_MODE = os.environ.get("UNDERWRITING_EXPORT_MODE", "otlp").strip().lower()
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "ai-underwriting-pipeline")


def _resource() -> Resource:
    # service.name is also read from OTEL_SERVICE_NAME by the SDK; setting it explicitly
    # keeps the console mode self-describing when the env var is absent.
    attrs = {"service.name": SERVICE_NAME}
    return Resource.create(attrs)


def _install_otlp(provider: TracerProvider) -> list[str]:
    """Attach OTLP exporters. Protocol follows OTEL_EXPORTER_OTLP_PROTOCOL."""
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").lower()
    installed = []

    if protocol.startswith("grpc"):
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    installed.append(f"otlp-traces({protocol})")

    # Metrics and logs are best-effort: Agent Observability's OTLP surface accepts
    # traces, and a missing metrics/logs endpoint should not stop the pipeline running.
    #
    # Honour OTEL_METRICS_EXPORTER / OTEL_LOGS_EXPORTER = "none". Without this the SDK
    # happily builds an exporter pointed at the default localhost:4318, and with no
    # collector in the container every export cycle logs a ConnectionError traceback --
    # thousands of lines that bury the actual startup failure.
    if os.environ.get("OTEL_METRICS_EXPORTER", "").strip().lower() == "none":
        installed.append("otlp-metrics(disabled)")
    else:
        try:
            metrics.set_meter_provider(
                MeterProvider(
                    resource=_resource(),
                    metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
                )
            )
            installed.append("otlp-metrics")
        except Exception as exc:  # pragma: no cover - depends on deployment
            print(f"[telemetry] metrics exporter not installed: {exc}", file=sys.stderr)

    if os.environ.get("OTEL_LOGS_EXPORTER", "").strip().lower() == "none":
        installed.append("otlp-logs(disabled)")
        return installed

    try:
        logger_provider = LoggerProvider(resource=_resource())
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter())
        )
        _logs.set_logger_provider(logger_provider)
        _events.set_event_logger_provider(EventLoggerProvider(logger_provider))
        installed.append("otlp-logs+events")
    except Exception as exc:  # pragma: no cover
        print(f"[telemetry] logs exporter not installed: {exc}", file=sys.stderr)

    return installed


def init_telemetry() -> dict:
    """Configure providers and instrument LangChain. Returns a description for logging."""
    provider = TracerProvider(resource=_resource())
    installed: list[str] = []

    if EXPORT_MODE in ("otlp", "both"):
        installed += _install_otlp(provider)

    if EXPORT_MODE in ("console", "both"):
        # Two decisions here, both required for the spans to be recoverable downstream.
        #
        # 1. SimpleSpanProcessor, not Batch -- one span emitted as it ends, rather than a
        #    JSON array of many.
        # 2. A COMPACT single-line formatter. ConsoleSpanExporter's default pretty-prints
        #    with indentation, and CloudWatch line-fragments multiline output: each span
        #    then arrives as dozens of separate log records that cannot be reassembled,
        #    and a search for any given attribute matches only the fragment containing it.
        #    Measured before this fix: a record matched `gen_ai.usage.input_tokens` but the
        #    same span's `trace_id`, `span_id` and `gen_ai.request.model` were in other
        #    records. One span per line is what makes recovery possible at all.
        provider.add_span_processor(
            SimpleSpanProcessor(
                ConsoleSpanExporter(
                    out=sys.stdout,
                    formatter=lambda span: span.to_json(indent=None) + "\n",
                )
            )
        )
        installed.append("console-spans(stdout->cloudwatch, one-line)")

    trace.set_tracer_provider(provider)

    # GenAI instrumentation. This is what produces gen_ai.* attributes, which is the
    # difference between a span that proves an invocation happened and a span that
    # carries model identity and token counts.
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor

    LangchainInstrumentor().instrument()
    installed.append("langchain-genai-instrumentation")

    info = {"export_mode": EXPORT_MODE, "service_name": SERVICE_NAME, "installed": installed}
    print(f"[telemetry] {info}", file=sys.stderr)
    return info


def tracer():
    return trace.get_tracer("ai.underwriting.pipeline")
