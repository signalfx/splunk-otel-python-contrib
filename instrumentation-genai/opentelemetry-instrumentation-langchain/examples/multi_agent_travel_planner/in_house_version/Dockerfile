# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the local OpenTelemetry packages first
# Note: Build context should be the repository root for this to work
COPY util/opentelemetry-util-genai /app/util/opentelemetry-util-genai
COPY util/opentelemetry-util-genai-evals /app/util/opentelemetry-util-genai-evals
COPY util/opentelemetry-util-genai-evals-deepeval /app/util/opentelemetry-util-genai-evals-deepeval
COPY util/opentelemetry-util-genai-emitters-splunk /app/util/opentelemetry-util-genai-emitters-splunk

# Install the local OpenTelemetry packages
RUN pip install --no-cache-dir -e /app/util/opentelemetry-util-genai \
    && pip install --no-cache-dir -e /app/util/opentelemetry-util-genai-evals \
    && pip install --no-cache-dir -e /app/util/opentelemetry-util-genai-evals-deepeval \
    && pip install --no-cache-dir -e /app/util/opentelemetry-util-genai-emitters-splunk

# Copy requirements file (for other dependencies)
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/in_house_version/main_in_house.py main.py
# Note: .env.example is optional, only copy if needed
# COPY instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/.env.example .env.example

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (adjust if your app uses a different port)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "main.py"]