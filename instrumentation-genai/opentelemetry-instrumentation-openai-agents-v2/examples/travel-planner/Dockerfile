FROM python:3.12-slim

WORKDIR /app

# Install git and build tools for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the OpenAI Agents instrumentation package and travel-planner example
COPY instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2 /app/opentelemetry-instrumentation-openai-agents-v2

# Copy the util-genai package (dependency of instrumentation)
COPY util/opentelemetry-util-genai /app/opentelemetry-util-genai

# Set working directory to travel-planner example
WORKDIR /app/opentelemetry-instrumentation-openai-agents-v2/examples/travel-planner

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install local util-genai package (provides opentelemetry.util.genai namespace)
# Use pip install with package name alias to satisfy 'opentelemetry-util-genai' dependency
RUN pip install --no-cache-dir /app/opentelemetry-util-genai && \
    pip install --no-cache-dir opentelemetry-util-genai==0.1.0 2>/dev/null || true

# Install local OpenAI Agents instrumentation package with instruments extra (includes openai-agents)
# Use --no-deps first, then install deps separately to avoid overwriting local util-genai
RUN pip install --no-cache-dir --no-deps "/app/opentelemetry-instrumentation-openai-agents-v2[instruments]" && \
    pip install --no-cache-dir openai-agents>=0.3.3

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor; print('✓ OpenAI Agents instrumentation available')" && \
    python3 -c "from opentelemetry.util.genai.handler import TelemetryHandler; print('✓ splunk-otel-util-genai available')" && \
    python3 -c "from util import OAuth2TokenManager; print('✓ OAuth2TokenManager available')"

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1

# Run the travel planner example with manual instrumentation
CMD ["python3", "main.py", "--manual-instrumentation"]
