FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only required directories for package installation
COPY util/ /app/util/
COPY instrumentation-genai/ /app/instrumentation-genai/

RUN echo "=== Contents of /app ===" && \
    ls -la /app/ && \
    echo "=== Checking for util directory ===" && \
    ls -la /app/util/ || echo "ERROR: util dir not found at /app/util/" && \
    echo "=== Checking for opentelemetry-util-genai ===" && \
    ls -la /app/util/opentelemetry-util-genai/ || echo "ERROR: opentelemetry-util-genai not found" && \
    echo "=== Checking pyproject.toml ===" && \
    test -f /app/util/opentelemetry-util-genai/pyproject.toml && echo "pyproject.toml exists" || echo "ERROR: No pyproject.toml"

RUN python -m pip install --upgrade pip

RUN python -m pip install pre-commit rstcheck || true

# Set HOME to a writable directory BEFORE installing deepeval-related packages
# This allows deepeval to create .deepeval directory during import
ENV HOME=/tmp

RUN pip uninstall -y opentelemetry-util-genai || true && \
    pip install -e /app/util/opentelemetry-util-genai --no-deps

RUN pip uninstall -y opentelemetry-util-genai-emitters-splunk || true && \
    pip install -e /app/util/opentelemetry-util-genai-emitters-splunk --no-deps

RUN pip uninstall -y opentelemetry-util-genai-evals || true && \
    pip install -e /app/util/opentelemetry-util-genai-evals --no-deps

RUN pip uninstall -y opentelemetry-util-genai-evals-deepeval || true && \
    pip install -e /app/util/opentelemetry-util-genai-evals-deepeval

RUN pip uninstall -y opentelemetry-instrumentation-langchain || true && \
    pip install -e /app/instrumentation-genai/opentelemetry-instrumentation-langchain

RUN pip install -r /app/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/requirements.txt

WORKDIR /app/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner

# Pre-create .deepeval directory in the working directory to avoid permission errors
# This must be done AFTER setting WORKDIR since deepeval creates .deepeval in CWD
RUN mkdir -p .deepeval && \
    chmod 777 .deepeval && \
    python -c "import deepeval" || true

EXPOSE 8080
CMD ["python", "main.py"]
