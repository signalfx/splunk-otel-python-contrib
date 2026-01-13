FROM python:3.12-slim

WORKDIR /app

# Install git for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the CrewAI instrumentation package and zero-code example
COPY instrumentation-genai/opentelemetry-instrumentation-crewai /app/opentelemetry-instrumentation-crewai
# Copy the shared OAuth2 helper package (not part of core util-genai)
COPY util/opentelemetry-util-genai-test /app/opentelemetry-util-genai-test

# Set working directory to zero-code example
WORKDIR /app/opentelemetry-instrumentation-crewai/examples/zero-code

# Install shared OAuth2 helper from local source (ensures availability even if not on PyPI)
RUN pip install --no-cache-dir /app/opentelemetry-util-genai-test

# Install Python dependencies (including genai utils from PyPI)
RUN pip install --no-cache-dir -r requirements.txt

# Install local CrewAI instrumentation package with instruments extras
RUN pip install --no-cache-dir /app/opentelemetry-instrumentation-crewai[instruments]

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.crewai import CrewAIInstrumentor; print('✓ CrewAI instrumentation available')" && \
    python3 -c "from opentelemetry.util.genai.handler import get_telemetry_handler; print('✓ GenAI handler available (from PyPI)')" && \
    python3 -c "from opentelemetry.util.oauth2_token_manager import OAuth2TokenManager; print('✓ OAuth2TokenManager available')" && \
    python3 -c "import opentelemetry.instrumentation; print('✓ OpenTelemetry instrumentation available')"

# Copy and make the run script executable
COPY instrumentation-genai/opentelemetry-instrumentation-crewai/examples/zero-code/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true

# Use wrapper script for proper telemetry flushing
CMD ["/app/run.sh"]
