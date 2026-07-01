FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Splunk instrumentation packages (pinned to latest PyPI)
RUN pip install --no-cache-dir \
    splunk-otel-util-genai==0.1.14 \
    splunk-otel-instrumentation-langchain==0.1.14 \
    splunk-otel-genai-emitters-splunk==0.1.8

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY main.py .

RUN useradd --create-home app && chown -R app:app /app
USER app

CMD ["opentelemetry-instrument", "python", "main.py"]
