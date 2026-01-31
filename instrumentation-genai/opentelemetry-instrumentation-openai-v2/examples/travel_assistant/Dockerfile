FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create token cache directory with proper permissions
RUN mkdir -p /tmp && chmod 755 /tmp

# Copy requirements first for better caching
COPY instrumentation-genai/opentelemetry-instrumentation-openai-v2/examples/travel_assistant/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local packages source code (util-genai + instrumentation)
COPY util/opentelemetry-util-genai /tmp/opentelemetry-util-genai
COPY instrumentation-genai/opentelemetry-instrumentation-openai-v2 /tmp/opentelemetry-instrumentation-openai-v2

# Install local packages (this will also install their dependencies)
# Then force reinstall without deps to ensure local code takes precedence
RUN pip install /tmp/opentelemetry-util-genai
RUN pip install /tmp/opentelemetry-instrumentation-openai-v2
# Force reinstall local packages to override any cached versions
RUN pip install --force-reinstall --no-deps /tmp/opentelemetry-util-genai
RUN pip install --force-reinstall --no-deps /tmp/opentelemetry-instrumentation-openai-v2

# Copy application code (main.py and util folder)
COPY instrumentation-genai/opentelemetry-instrumentation-openai-v2/examples/travel_assistant/main.py .
COPY instrumentation-genai/opentelemetry-instrumentation-openai-v2/examples/travel_assistant/util ./util

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
ENTRYPOINT ["python", "main.py"]
