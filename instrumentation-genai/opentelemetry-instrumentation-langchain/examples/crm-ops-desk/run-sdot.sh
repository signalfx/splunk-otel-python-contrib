#!/usr/bin/env bash
# Run CRM Ops Desk demo with SDOT (Splunk Distribution of OpenTelemetry) instrumentation
# Sends traces to local OTel Collector via gRPC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Credentials ──────────────────────────────────────────────────
# OpenAI (if not in .env)
#   Required variables:
#     OPENAI_API_KEY="sk-..."

# SSL — corporate proxy CA bundle (optional)
[[ -f ~/.corporate-certs/env.sh ]] && source ~/.corporate-certs/env.sh

# ── OTel / SDOT configuration ───────────────────────────────────
export OTEL_SERVICE_NAME="crm-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_METRICS_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE="DELTA"
export OTEL_LOGS_EXPORTER="otlp"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=crm-demo"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE="SPAN_AND_EVENT"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,splunk"

# ── App .env (OpenAI) ───────────────────────────────────────────
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# ── Run ──────────────────────────────────────────────────────────
echo "=== SDOT-instrumented CRM Ops Desk (LangGraph) ==="
echo "Service:    $OTEL_SERVICE_NAME"
echo "Collector:  $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "=================================================="

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Run with opentelemetry-instrument to auto-discover SDOT instrumentors
exec opentelemetry-instrument python main.py "$@"
