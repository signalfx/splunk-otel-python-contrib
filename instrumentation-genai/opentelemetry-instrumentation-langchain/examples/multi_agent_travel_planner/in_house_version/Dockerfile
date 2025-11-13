# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Splunk OpenTelemetry packages from PyPI
RUN pip install --no-cache-dir \
    splunk-otel-util-genai==0.1.4 \
    splunk-otel-genai-emitters-splunk==0.1.4 \
    splunk-otel-util-genai-evals==0.1.4 \
    splunk-otel-genai-evals-deepeval==0.1.6 \
    splunk-otel-instrumentation-langchain==0.1.4

# Copy requirements file (for other dependencies like langchain, langgraph, etc.)
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