FROM python:3.12-slim

# Define Traceloop SDK version as build arg (also used for opentelemetry-instrumentation-* packages)
ARG TRACELOOP_VERSION=0.47.5

WORKDIR /app

# Install git for pip dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy required packages (build context is repo root)
COPY util/opentelemetry-util-genai-traceloop-translator /app/opentelemetry-util-genai-traceloop-translator
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner /app/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner

WORKDIR /app/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/traceloop

# Install Python dependencies (excluding local -e packages and git+ssh dependencies)
RUN pip install --no-cache-dir -r requirements.traceloop.txt

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.util.genai.handler import get_telemetry_handler; print('✓ GenAI handler available')" && \
    python3 -c "from opentelemetry.util.genai.evals import create_evaluation_manager; print('✓ Evaluation manager available')" && \
    python3 -c "import opentelemetry.util.genai.emitters.splunk; print('✓ Splunk emitters available')" && \
    python3 -c "import opentelemetry.util.evaluator.deepeval; print('✓ Deepeval evaluator module available')" && \
    python3 -c "import deepeval; print('✓ Deepeval SDK installed')" && \
    python3 -c "from opentelemetry.util.genai.traceloop import enable_traceloop_translator; print('✓ Traceloop translator available')"

# Make the script executable
RUN chmod +x main_traceloop.py

# Set default environment variables
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    PYTHONUNBUFFERED=1

# Run the Traceloop version
CMD ["python3", "main_traceloop.py"]