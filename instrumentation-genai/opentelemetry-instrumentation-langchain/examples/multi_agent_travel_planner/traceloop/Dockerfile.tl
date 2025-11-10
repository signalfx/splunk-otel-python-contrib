# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire instrumentation-genai and util directories to maintain package paths
# Build context is at repo root to access both instrumentation-genai/ and util/
# DO NOT INSTALL splunk-otel-instrumentation-langchain; If present uninstall it
COPY instrumentation-genai/opentelemetry-instrumentation-langchain /app/opentelemetry-instrumentation-langchain
COPY util/opentelemetry-util-genai /app/opentelemetry-util-genai
COPY util/opentelemetry-util-genai-traceloop-translator /app/opentelemetry-util-genai-traceloop-translator
COPY util/opentelemetry-util-genai-evals /app/opentelemetry-util-genai-evals
COPY util/opentelemetry-util-genai-evals-deepeval /app/opentelemetry-util-genai-evals-deepeval
COPY util/opentelemetry-util-genai-emitters-splunk /app/opentelemetry-util-genai-emitters-splunk

# Set working directory to the traceloop example
WORKDIR /app/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/traceloop

# Install Python dependencies from requirements.traceloop.txt (excluding local -e packages and git+ssh)
# First, create a temporary requirements file without the local editable packages and SSH git dependencies
RUN grep -v "^-e" requirements.traceloop.txt | grep -v "git+ssh" > /tmp/requirements_external.txt && \
    pip install --no-cache-dir -r /tmp/requirements_external.txt && \
    rm /tmp/requirements_external.txt

# Install the local packages in editable mode
# The Traceloop translator will enable zero-code instrumentation via .pth file
RUN cd /app/opentelemetry-util-genai && \
    pip install --no-cache-dir --no-deps -e . && \
    cd /app/opentelemetry-util-genai-evals && \
    pip install --no-cache-dir --no-deps -e . && \
    cd /app/opentelemetry-util-genai-evals-deepeval && \
    pip install --no-cache-dir --no-deps -e . && \
    cd /app/opentelemetry-util-genai-emitters-splunk && \
    pip install --no-cache-dir --no-deps -e . && \
    cd /app/opentelemetry-util-genai-traceloop-translator && \
    pip install --no-cache-dir --no-deps -e .

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.util.genai.handler import get_telemetry_handler; print('✓ GenAI handler available')" && \
    python3 -c "from opentelemetry.util.genai.evals import create_evaluation_manager; print('✓ Evaluation manager available')" && \
    python3 -c "import opentelemetry.util.genai.emitters.splunk; print('✓ Splunk emitters available')" && \
    python3 -c "import opentelemetry.util.evaluator.deepeval; print('✓ Deepeval evaluator module available')" && \
    python3 -c "import deepeval; print('✓ Deepeval SDK installed')" && \
    python3 -c "from opentelemetry.util.genai.traceloop import enable_traceloop_translator; print('✓ Traceloop translator available')"

# Make the script executable
RUN chmod +x main_traceloop.py

# Set default environment variables for OpenTelemetry
ENV OTEL_PYTHON_LOG_CORRELATION=true \
    OTEL_PYTHON_LOG_LEVEL=info \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    PYTHONUNBUFFERED=1

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Run the Traceloop version
CMD ["python3", "main_traceloop.py"]

