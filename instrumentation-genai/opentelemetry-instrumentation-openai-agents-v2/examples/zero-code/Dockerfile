FROM python:3.12-slim

WORKDIR /app

# Install git for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the OpenAI Agents instrumentation package and zero-code example
COPY instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2 /app/opentelemetry-instrumentation-openai-agents-v2

# Set working directory to zero-code example
WORKDIR /app/opentelemetry-instrumentation-openai-agents-v2/examples/zero-code

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install local OpenAI Agents instrumentation package
RUN pip install --no-cache-dir /app/opentelemetry-instrumentation-openai-agents-v2

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor; print('✓ OpenAI Agents instrumentation available')" && \
    python3 -c "import opentelemetry.instrumentation; print('✓ OpenTelemetry instrumentation available')"

# Copy and make the run script executable
COPY instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/zero-code/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    PYTHONUNBUFFERED=1

# Use wrapper script for proper telemetry flushing
CMD ["/app/run.sh"]
