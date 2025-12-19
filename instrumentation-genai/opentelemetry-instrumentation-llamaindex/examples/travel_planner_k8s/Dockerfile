FROM python:3.12-slim

WORKDIR /app

# Disable telemetry during build to avoid connection errors
ENV OTEL_SDK_DISABLED=true

# Install dependencies
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/travel_planner_k8s/requirements.txt .
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt

# Copy and install local instrumentation package
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex /tmp/instrumentation-llamaindex/
RUN pip install --no-cache-dir /tmp/instrumentation-llamaindex && \
    rm -rf /tmp/instrumentation-llamaindex

# Copy application code
COPY instrumentation-genai/opentelemetry-instrumentation-llamaindex/examples/travel_planner_k8s/main_server.py .

# Expose port
EXPOSE 8080

# Run the server
CMD ["python", "main_server.py"]
