# Multi-Agent Travel Planner Example

This example demonstrates a multi-agent travel planning workflow using the
OpenAI Agents SDK with OpenTelemetry instrumentation. It leverages Splunk
distribution of `opentelemetry-util-genai` for producing telemetry in
semantic convention.

## Agents

- **Flight Specialist**: Searches for flight options
- **Hotel Specialist**: Recommends accommodations
- **Activity Specialist**: Curates local activities and experiences
- **Travel Coordinator**: Orchestrates and synthesizes the final itinerary

## Instrumentation Modes

This example supports two instrumentation modes controlled by `--manual-instrumentation`:

**Manual Instrumentation:**

```bash
python main.py --manual-instrumentation
```

**Zero-Code Instrumentation:**

```bash
opentelemetry-instrument python main.py
```

## Prerequisites

- Python 3.10+
- Access to an LLM provider (OpenAI API key or OAuth2 credentials)

Install the Splunk Distribution of OpenTelemetry packages:

```bash
# Core instrumentation packages
pip install splunk-otel-instrumentation-openai-agents-v2
pip install splunk-otel-util-genai

# Splunk-specific emitters (required for Splunk Observability Cloud)
pip install splunk-otel-genai-emitters-splunk

# Or install all at once using requirements.txt
pip install -r requirements.txt
```

## Setup

1. **Change to the examples directory:**

```bash
cd instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/travel-planner
```

2. **Create the virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

**For local development** (if you want to test unreleased changes):

```bash
pip install splunk-otel-util-genai
pip install splunk-otel-genai-emitters-splunk
pip install splunk-otel-instrumentation-openai-agents-v2
```

4. **Create environment variable configuration:**

```bash
cp .env.example .env
```

5. **Set the required environment variables in `.env`:**

**LLM Credentials (Option 1 - OpenAI):**

```bash
OPENAI_API_KEY=your-openai-api-key
```

**LLM Credentials (Option 2 - OAuth2):**

```bash
LLM_CLIENT_ID=your-client-id
LLM_CLIENT_SECRET=your-client-secret
LLM_TOKEN_URL=https://your-identity-provider/oauth2/token
LLM_BASE_URL=https://your-llm-gateway/openai/deployments
OPENAI_MODEL_NAME=gpt-4o-mini
```

**OpenTelemetry Configuration:**

```bash
OTEL_SERVICE_NAME=travel-planner
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=demo
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_LOGS_EXPORTER=otlp
```

**GenAI Instrumentation:**

```bash
# Emitters configuration
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk

# Message content capture
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
```

**(Optional) LLM-as-a-Judge Evaluations:**

```bash
# Enable evaluation results aggregation
OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true

# Evaluation emitter configuration
OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults

# Configure evaluators (e.g., DeepEval metrics)
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=Deepeval(LLMInvocation(bias,toxicity))
```

6. **Start the OpenTelemetry Collector:**

```bash
docker run -p 4317:4317 otel/opentelemetry-collector:latest
```

7. **Load environment variables and run:**

```bash
source .env
python main.py --manual-instrumentation
```

## Expected Trace Structure

```
gen_ai.workflow (Travel Planner)
├── invoke_agent (Flight Specialist)
│   ├── chat (OpenAI)
│   └── execute_tool (search_flights)
├── invoke_agent (Hotel Specialist)
│   ├── chat (OpenAI)
│   └── execute_tool (search_hotels)
├── invoke_agent (Activity Specialist)
│   ├── chat (OpenAI)
│   └── execute_tool (search_activities)
└── invoke_agent (Travel Coordinator)
    └── chat (OpenAI)
```

## Docker

Build and run with Docker:

```bash
# From repository root
docker build -f instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/examples/travel-planner/Dockerfile -t openai-agents-travel-planner .
docker run --env-file .env openai-agents-travel-planner
```

## Kubernetes

Create Secrets:

```bash
kubectl create secret generic llm-credentials \
  --from-literal=client-id=your-client-id \
  --from-literal=client-secret=your-client-secret \
  --from-literal=token-url=https://your-idp/oauth2/token \
  --from-literal=base-url=https://your-llm-gateway/openai/deployments
```

Deploy CronJob:

```bash
kubectl apply -f cronjob.yaml
```

## Project Structure

```
travel-planner/
├── main.py                      # Multi-agent travel planner
├── util/
│   ├── __init__.py
│   └── oauth2_token_manager.py  # OAuth2 token management
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── Dockerfile                   # Container build
├── cronjob.yaml                 # Kubernetes CronJob spec
└── README.md                    # This file
```

## Environment Variables Reference

**LLM Configuration:**

| Variable | Description | Required |
|---|---|---|
| `LLM_CLIENT_ID` | OAuth2 client ID | Yes (OAuth2) |
| `LLM_CLIENT_SECRET` | OAuth2 client secret | Yes (OAuth2) |
| `LLM_TOKEN_URL` | OAuth2 token endpoint | Yes (OAuth2) |
| `LLM_BASE_URL` | LLM gateway base URL | Yes (OAuth2) |
| `OPENAI_API_KEY` | OpenAI API key | Yes (OpenAI) |

**GenAI Instrumentation:**

| Variable | Description | Default |
|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | GenAI emitters | `span` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture message content | `false` |

## Related Documentation

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [Splunk Observability for AI](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/set-up-observability-for-ai)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
