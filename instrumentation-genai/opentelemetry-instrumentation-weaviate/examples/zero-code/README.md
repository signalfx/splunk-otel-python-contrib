# Weaviate Zero-Code Instrumentation

This guide shows how to run the zero-code Weaviate instrumentation example in this
repo using OpenTelemetry auto-instrumentation.

## Step-by-step

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r instrumentation-genai/opentelemetry-instrumentation-weaviate/examples/zero-code/requirements.txt
pip install -e instrumentation-genai/opentelemetry-instrumentation-weaviate
```

2. Start a local Weaviate instance:

```bash
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
```

3. Configure OpenTelemetry (update endpoint to your collector):

```bash
export OTEL_SERVICE_NAME="weaviate-zero-code-example"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_TRACES_EXPORTER="otlp"
```

4. Run the sample app with auto-instrumentation:

```bash
opentelemetry-instrument python instrumentation-genai/opentelemetry-instrumentation-weaviate/examples/zero-code/main.py
```

## What this uses

- Sample app: `instrumentation-genai/opentelemetry-instrumentation-weaviate/examples/zero-code/main.py`
- Instrumentation package: `instrumentation-genai/opentelemetry-instrumentation-weaviate`
