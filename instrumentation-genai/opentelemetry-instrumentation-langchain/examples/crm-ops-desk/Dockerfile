FROM python:3.12-slim

WORKDIR /build

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# ── Install SDOT packages from local source ────────────────
COPY util/opentelemetry-util-genai /build/util/opentelemetry-util-genai
RUN pip install --no-cache-dir /build/util/opentelemetry-util-genai

COPY util/opentelemetry-util-genai-emitters-splunk /build/util/opentelemetry-util-genai-emitters-splunk
RUN pip install --no-cache-dir /build/util/opentelemetry-util-genai-emitters-splunk

COPY instrumentation-genai/opentelemetry-instrumentation-langchain /build/instrumentation-genai/opentelemetry-instrumentation-langchain
RUN pip install --no-cache-dir "/build/instrumentation-genai/opentelemetry-instrumentation-langchain[instruments]"

# OTLP exporter for gRPC
RUN pip install --no-cache-dir opentelemetry-exporter-otlp-proto-grpc

# ── Install CRM app dependencies ──────────────────────────
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/crm-ops-desk/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt

# ── Copy application code ─────────────────────────────────
WORKDIR /app
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/crm-ops-desk /app

# Remove local .env (env vars injected via K8s configmap/secret)
RUN rm -f /app/.env /app/.env.example

# ── Verify instrumentation ────────────────────────────────
RUN python -c "from opentelemetry.instrumentation.langchain import LangChainInstrumentor; print('OK: LangChain instrumentation')" && \
    python -c "from opentelemetry.util.genai.handler import TelemetryHandler; print('OK: splunk-otel-util-genai')"

# ── Cleanup build artefacts ───────────────────────────────
RUN rm -rf /build

# Run with zero-code instrumentation (opentelemetry-instrument)
ENTRYPOINT ["opentelemetry-instrument", "python"]
CMD ["main.py", "--index", "0"]
