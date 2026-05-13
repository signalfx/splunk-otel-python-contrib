FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/single-agent/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local packages source code
COPY util/opentelemetry-util-genai /tmp/opentelemetry-util-genai
COPY util/opentelemetry-util-genai-evals /tmp/opentelemetry-util-genai-evals
COPY util/opentelemetry-util-genai-evals-deepeval /tmp/opentelemetry-util-genai-evals-deepeval
COPY util/opentelemetry-util-genai-emitters-splunk /tmp/opentelemetry-util-genai-emitters-splunk
COPY instrumentation-genai/opentelemetry-instrumentation-vertexai /tmp/opentelemetry-instrumentation-vertexai

# Install local packages with --no-deps to avoid version conflicts
# between local dev packages (requirements.txt covers external deps)
RUN pip install --no-deps /tmp/opentelemetry-util-genai-evals \
    /tmp/opentelemetry-util-genai-evals-deepeval \
    /tmp/opentelemetry-util-genai-emitters-splunk \
    /tmp/opentelemetry-instrumentation-vertexai \
    && pip install --no-deps /tmp/opentelemetry-util-genai

# Copy application code
COPY instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/single-agent/main.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "main.py"]
