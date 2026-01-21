FROM python:3.12-slim

WORKDIR /app

# Install git and build tools for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the CrewAI instrumentation package
COPY instrumentation-genai/opentelemetry-instrumentation-crewai /app/opentelemetry-instrumentation-crewai

# Set working directory to example
WORKDIR /app/opentelemetry-instrumentation-crewai/examples

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install local CrewAI instrumentation package with instruments extras
RUN pip install --no-cache-dir /app/opentelemetry-instrumentation-crewai[instruments]

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.crewai import CrewAIInstrumentor; print('✓ CrewAI instrumentation available')" && \
    python3 -c "from opentelemetry.util.genai.handler import get_telemetry_handler; print('✓ GenAI handler available')" && \
    python3 -c "from util import OAuth2TokenManager; print('✓ OAuth2TokenManager available')" && \
    python3 -c "import opentelemetry.instrumentation; print('✓ OpenTelemetry instrumentation available')"

# Copy the run script
COPY instrumentation-genai/opentelemetry-instrumentation-crewai/examples/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true \
    OTEL_MANUAL_INSTRUMENTATION=true

# Use run.sh which handles both manual and zero-code modes
CMD ["/app/run.sh"]

