FROM python:3.12-slim

WORKDIR /app

# Install git and build tools for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the CrewAI instrumentation package and manual example
COPY instrumentation-genai/opentelemetry-instrumentation-crewai /app/opentelemetry-instrumentation-crewai

# Set working directory to manual example
WORKDIR /app/opentelemetry-instrumentation-crewai/examples/manual

# Install Python dependencies (including genai utils from PyPI)
RUN pip install --no-cache-dir -r requirements.txt

# Install local CrewAI instrumentation package with instruments extras
RUN pip install --no-cache-dir /app/opentelemetry-instrumentation-crewai[instruments]

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.crewai import CrewAIInstrumentor; print('✓ CrewAI instrumentation available')" && \
    python3 -c "from opentelemetry.util.genai.handler import get_telemetry_handler; print('✓ GenAI handler available (from PyPI)')" && \
    python3 -c "from util import OAuth2TokenManager; print('✓ OAuth2TokenManager available')"

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1 \
    CREWAI_DISABLE_TELEMETRY=true

# Run the customer support example
CMD ["python3", "customer_support.py"]
