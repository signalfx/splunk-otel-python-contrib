FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the example application
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner /app/multi_agent_travel_planner

WORKDIR /app/multi_agent_travel_planner

RUN python -m pip install --upgrade pip

# Set HOME to a writable directory BEFORE installing deepeval-related packages
ENV HOME=/tmp

# Install application dependencies (includes Splunk OpenTelemetry GenAI packages)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-create .deepeval directory to avoid permission errors
RUN mkdir -p .deepeval && \
    chmod 777 .deepeval && \
    python -c "import deepeval" || true

# Verify packages are installed correctly
RUN python3 -c "from opentelemetry.instrumentation.langchain import LangChainInstrumentor; print('✓ LangChain instrumentation available')" && \
    python3 -c "from opentelemetry.util.genai.handler import TelemetryHandler; print('✓ splunk-otel-util-genai available')"

EXPOSE 8080
CMD ["python", "main.py", "--manual-instrumentation"]
