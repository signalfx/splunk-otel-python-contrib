# VertexAI Single-Agent Weather Example

A ReAct-style agentic loop using the native VertexAI SDK (GenerativeModel + tool calling) with automatic OpenTelemetry instrumentation via `VertexAIInstrumentor`.

## Overview

The agent:

1. Receives a weather query (e.g. "What is the weather in San Francisco?")
2. Calls Gemini with a `get_weather` tool declaration
3. Dispatches the tool call to the [Open-Meteo API](https://open-meteo.com/) (no API key required)
4. Returns the tool result to Gemini for a final natural-language response

## Prerequisites

- Python 3.10+
- GCP project with Vertex AI API enabled
- Application Default Credentials (ADC) or a service-account JSON

## Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-vertexai/examples/single-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create environment configuration:

```bash
cp .env.example .env
# Edit .env with your GCP project ID and OTLP endpoint
```

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | — |
| `GOOGLE_CLOUD_LOCATION` | GCP region | `us-central1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service-account JSON (if not using ADC) | — |

### OpenTelemetry Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol | `grpc` |
| `OTEL_SERVICE_NAME` | Service name for telemetry | `vertexai-single-agent` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitters to use | `span_metric_event` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompts/completions | `true` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | Content capture mode | `SPAN_AND_EVENT` |

## Running

```bash
source .venv/bin/activate
python main.py                       # default: San Francisco
python main.py --city "Tokyo"        # custom city
```

## Sample Telemetry

Sample output from a single run with `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event`.

### Traces

```
Service Name                      Latency   Span Name
opentelemetry-python-vertexai     1.09s     chat gemini-2.5-flash-lite   ← initial query + tool call
opentelemetry-python-vertexai     528.48ms  chat gemini-2.5-flash-lite   ← tool result → final answer
```

Span attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`,
`server.address`, `server.port`.

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gen_ai.client.token.usage` | Histogram | Input/output token counts per request |
| `gen_ai.client.operation.duration` | Histogram | LLM call latency in seconds |

Dimensions: `gen_ai.token.type`, `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.operation.name`, `gen_ai.framework`, `gen_ai.provider.name`.

### Log Events

When `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT`:

- **Event**: `gen_ai.client.inference.operation.details`
- **Body**: full `gen_ai.input.messages` and `gen_ai.output.messages` as JSON
- Correlated to parent span via `trace_id` / `span_id`

## Project Structure

```
single-agent/
├── main.py              # Agent loop with manual OTel setup
├── requirements.txt     # Pinned dependencies (app + SDOT packages)
├── .env.example         # Environment variable template
├── Dockerfile           # Container build
├── cronjob.yaml         # Kubernetes CronJob spec
└── README.md            # This file
```

## Kubernetes Deployment

```bash
kubectl apply -f cronjob.yaml
```

## Related Documentation

- [Vertex AI Python SDK](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk)
- [Splunk Observability for AI](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/set-up-observability-for-ai)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
