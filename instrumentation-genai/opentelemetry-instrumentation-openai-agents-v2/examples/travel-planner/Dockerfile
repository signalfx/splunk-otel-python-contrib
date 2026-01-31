FROM python:3.12-slim

WORKDIR /app

# Install git and build tools for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the travel-planner example
COPY instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/travel-planner /app/travel-planner

# Set working directory to travel-planner example
WORKDIR /app/travel-planner

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies (includes Splunk OpenTelemetry GenAI packages)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-create .deepeval directory to avoid permission errors
RUN mkdir -p .deepeval && \
    chmod 777 .deepeval && \
    python -c "import deepeval" || true

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor; print('✓ OpenAI Agents instrumentation available')" && \
    python3 -c "from util import OAuth2TokenManager; print('✓ OAuth2TokenManager available')"

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1

# Run the travel planner example with manual instrumentation
CMD ["python3", "main.py", "--manual-instrumentation"]
