# CrewAI Manual Instrumentation Example

This example demonstrates **manual instrumentation** of CrewAI applications with OpenTelemetry, including OAuth2 LLM authentication and optional GenAI evaluations via DeepEval.

## Prerequisites

1. **Python 3.9+**
2. **LLM Access** - OAuth2 credentials for your LLM provider (or OpenAI API key)
3. **OTel Collector** - For receiving telemetry data
4. **Splunk Observability Cloud** (optional) - For viewing traces, metrics, and evaluations

## Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples/manual

# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install CrewAI instrumentation (from local source)
pip install -e ../../[instruments]

# 4. Configure environment variables
cp env.example .env
# Edit .env with your credentials
```

## Environment Variables

### Required: LLM Configuration

```bash
# OAuth2 LLM Provider
LLM_CLIENT_ID=<your-oauth2-client-id>
LLM_CLIENT_SECRET=<your-oauth2-client-secret>
LLM_TOKEN_URL=https://<your-identity-provider>/oauth2/token
LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments
LLM_APP_KEY=<your-app-key>  # Optional, required by some providers
```

### Required: OpenTelemetry Configuration

```bash
# Service Identity
OTEL_SERVICE_NAME=crewai-demo-app
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=demo

# OTLP Exporter
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Logs
OTEL_LOGS_EXPORTER=otlp
OTEL_PYTHON_LOG_CORRELATION=true
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true

# Metrics
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
```

### Required: GenAI Instrumentation

```bash
# Emitters (span_metric_event for full telemetry, splunk for Splunk-specific features)
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk

# Content Capture
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT

# Evaluations
OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true
OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults
```

### Optional: DeepEval Custom LLM-as-a-Judge

Route evaluations through your own LLM provider instead of OpenAI:

```bash
DEEPEVAL_LLM_BASE_URL=https://<your-llm-gateway>/openai/deployments/<model>
DEEPEVAL_LLM_MODEL=gpt-4o-mini
DEEPEVAL_LLM_PROVIDER=openai
DEEPEVAL_LLM_CLIENT_ID=<your-oauth2-client-id>
DEEPEVAL_LLM_CLIENT_SECRET=<your-oauth2-client-secret>
DEEPEVAL_LLM_TOKEN_URL=https://<your-identity-provider>/oauth2/token
DEEPEVAL_LLM_CLIENT_APP_NAME=<your-app-key>
DEEPEVAL_FILE_SYSTEM=READ_ONLY
```

### Optional: Debug Settings

```bash
OTEL_INSTRUMENTATION_GENAI_DEBUG=false
OTEL_GENAI_EVAL_DEBUG_SKIPS=false
OTEL_GENAI_EVAL_DEBUG_EACH=false
OTEL_INSTRUMENTATION_LANGCHAIN_DEBUG=false
```

### CrewAI Settings

```bash
# Disable CrewAI's built-in telemetry (recommended when using OTel)
CREWAI_DISABLE_TELEMETRY=true
```

## Running the Example

### 1. Start OTel Collector

```bash
# Using Docker
docker run -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector

# Or use otel-tui for local debugging
docker run -p 4317:4317 -p 4318:4318 --name otel-tui ymtdzzz/otel-tui:latest
```

### 2. Run the Example

```bash
# Load environment variables and run
source .env  # or use: set -a && source .env && set +a

python customer_support.py
```

### 3. Expected Console Output

```
[AUTH] Token obtained (length: 1234)

 Agent: Senior Support Representative starting task...
 Agent: Support Quality Assurance Specialist starting task...

[SUCCESS] Crew execution completed

====================================================================================================
METRICS OUTPUT BELOW - Look for gen_ai.agent.duration and gen_ai.workflow.duration
====================================================================================================

[FLUSH] Starting telemetry flush
[FLUSH] Flushing traces (timeout=30s)
[FLUSH] Flushing metrics (timeout=30s)
[FLUSH] Shutting down metrics provider
[FLUSH] Telemetry flush complete
```

## Expected Trace Structure

```
gen_ai.workflow crew (customer-support-crew)                    [2.5s]
├── gen_ai.step inquiry_resolution                              [1.8s]
│   └── invoke_agent Senior Support Representative              [1.7s]
│       ├── chat gpt-4o-mini                                    [0.8s]
│       │   ├── gen_ai.request.model: gpt-4o-mini
│       │   ├── gen_ai.usage.input_tokens: 245
│       │   ├── gen_ai.usage.output_tokens: 512
│       │   └── gen_ai.response.finish_reasons: ["stop"]
│       └── tool Read website content                           [0.5s]
│           └── GET https://docs.crewai.com/...
└── gen_ai.step quality_assurance_review                        [0.7s]
    └── invoke_agent Support Quality Assurance Specialist       [0.6s]
        └── chat gpt-4o-mini                                    [0.5s]
            ├── gen_ai.request.model: gpt-4o-mini
            ├── gen_ai.usage.input_tokens: 890
            └── gen_ai.usage.output_tokens: 256
```

## Span Attributes

| Attribute | Description |
|-----------|-------------|
| `gen_ai.operation.name` | Operation type (e.g., `chat`, `workflow`, `step`) |
| `gen_ai.request.model` | Model used for LLM calls |
| `gen_ai.system` | System identifier (`crewai`, `openai`) |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.agent.name` | CrewAI agent role |
| `gen_ai.workflow.name` | CrewAI crew name |

## Metrics Generated

| Metric | Description |
|--------|-------------|
| `gen_ai.workflow.duration` | Total crew execution time |
| `gen_ai.agent.duration` | Per-agent execution time |
| `gen_ai.client.token.usage` | Token counts per LLM call |
| `gen_ai.client.operation.duration` | LLM call latency |

## Project Structure

```
manual/
├── customer_support.py        # Main example with OAuth2 LLM
├── financial_assistant.py     # Financial trading crew example
├── researcher_writer_manager.py  # Hierarchical crew example
├── util/
│   ├── __init__.py
│   └── oauth2_token_manager.py  # OAuth2 token management
├── requirements.txt           # Python dependencies
├── env.example               # Environment variable template
├── Dockerfile                # Container build
├── cronjob.yaml              # Kubernetes CronJob spec
└── README.md                 # This file
```

## Troubleshooting

### OAuth2 Token Errors

```
ValueError: OAuth2 credentials required...
```

**Fix:** Ensure `LLM_CLIENT_ID`, `LLM_CLIENT_SECRET`, and `LLM_TOKEN_URL` are set.

### LLM 404 Errors

```
Error code: 404 - {'detail': 'Not Found'}
```

**Fix:** Check `LLM_BASE_URL` is correct and doesn't include the model name (model is appended automatically).

### No Traces Appearing

1. Verify OTel collector is running: `curl http://localhost:4317`
2. Check `OTEL_EXPORTER_OTLP_ENDPOINT` is correct
3. Enable debug: `OTEL_LOG_LEVEL=debug`

### Evaluations Not Showing

1. Ensure `OTEL_INSTRUMENTATION_GENAI_EMITTERS` includes `splunk`
2. Check DeepEval credentials if using custom LLM-as-a-Judge
3. Verify `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true`

## Kubernetes Deployment

See `cronjob.yaml` for a complete Kubernetes CronJob specification. Update the secret names and image as needed:

```bash
# Create secrets
kubectl create secret generic llm-credentials \
  --from-literal=client-id=<your-client-id> \
  --from-literal=client-secret=<your-client-secret> \
  --from-literal=token-url=<your-token-url> \
  --from-literal=base-url=<your-base-url> \
  --from-literal=app-key=<your-app-key>

# Apply CronJob
kubectl apply -f cronjob.yaml
```

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk Observability for AI - Setup Guide](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/set-up-observability-for-ai)
- [DeepEval Documentation](https://docs.confident-ai.com/)

