FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY util/ /app/util/
COPY instrumentation-genai/ /app/instrumentation-genai/

RUN python -m pip install --upgrade pip

ENV HOME=/tmp

# Install application requirements first
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/zero-code/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install local packages in editable mode - this will install their dependencies too
RUN pip install -e /app/util/opentelemetry-util-genai && \
    pip install -e /app/util/opentelemetry-util-genai-evals && \
    pip install -e /app/util/opentelemetry-util-genai-evals-deepeval && \
    pip install -e /app/util/opentelemetry-util-genai-emitters-splunk && \
    pip install -e /app/instrumentation-genai/opentelemetry-instrumentation-llamaindex

# Upgrade OpenTelemetry packages to resolve version conflicts
# This ensures we have compatible versions even if local packages pinned older ones
RUN pip install --upgrade --force-reinstall \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation \
    opentelemetry-semantic-conventions \
    opentelemetry-distro \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-exporter-otlp-proto-http

# Copy application code
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/zero-code /app/zero-code

WORKDIR /app/zero-code

EXPOSE 8080

# Default command runs the server
CMD ["opentelemetry-instrument", "python", "server.py"]
