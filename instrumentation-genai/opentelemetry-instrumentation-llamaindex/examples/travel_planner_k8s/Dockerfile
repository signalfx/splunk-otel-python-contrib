FROM python:3.12-slim

WORKDIR /app

# Disable telemetry during build to avoid connection errors
ENV OTEL_SDK_DISABLED=true

# Install dependencies
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/travel_planner_k8s/requirements.txt .
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt

# Copy and install local util packages (dependencies for instrumentation)
COPY util/opentelemetry-util-genai /tmp/util-genai/
COPY util/opentelemetry-util-genai-evals /tmp/util-genai-evals/
COPY util/opentelemetry-util-genai-evals-deepeval /tmp/util-genai-evals-deepeval/
COPY util/opentelemetry-util-genai-emitters-splunk /tmp/util-genai-emitters-splunk/
RUN pip install --no-cache-dir /tmp/util-genai/ /tmp/util-genai-evals/ /tmp/util-genai-evals-deepeval/ /tmp/util-genai-emitters-splunk/ && \
    rm -rf /tmp/util-*

# Copy and install local instrumentation package
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex /tmp/instrumentation-llamaindex/
RUN pip install --no-cache-dir /tmp/instrumentation-llamaindex && \
    rm -rf /tmp/instrumentation-llamaindex

# Copy shared util module (OAuth2TokenManager)
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/util /app/util/

# Copy application code
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/travel_planner_k8s/main_server.py .

# Expose port
EXPOSE 8080

# Run the server
CMD ["python", "main_server.py"]
