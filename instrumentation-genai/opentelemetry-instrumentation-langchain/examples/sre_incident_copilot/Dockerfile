FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the langchain instrumentation package (editable)
RUN pip install --no-cache-dir -e /app/instrumentation || true

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/artifacts

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "main.py", "--scenario", "scenario-001"]

