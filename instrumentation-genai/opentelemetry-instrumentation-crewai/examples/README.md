# CrewAI Financial Trading Assistant with OpenTelemetry

A multi-agent financial trading crew demonstrating OpenTelemetry instrumentation for CrewAI applications. This example supports both **manual** and **zero-code** instrumentation modes.

## Overview

The Financial Trading Crew consists of four specialized agents:

| Agent | Role | Responsibilities |
|-------|------|-----------------|
| **Data Analyst** | Market Analysis | Monitor market data, identify trends, predict movements |
| **Strategy Developer** | Strategy Creation | Develop trading strategies based on insights |
| **Trade Advisor** | Execution Planning | Suggest optimal trade execution methods |
| **Risk Advisor** | Risk Assessment | Evaluate risks and suggest mitigation strategies |

## Instrumentation Modes

This example supports two instrumentation modes, controlled by the `OTEL_MANUAL_INSTRUMENTATION` environment variable:

### Manual Instrumentation (Default)

```bash
# Explicit in-code instrumentation with full control
export OTEL_MANUAL_INSTRUMENTATION=true
python financial_assistant.py
```

**Benefits:**
- Full control over tracer and meter providers
- Custom processors and exporters
- Console output for local debugging

### Zero-Code Instrumentation

```bash
# Auto-instrumentation via opentelemetry-instrument
export OTEL_MANUAL_INSTRUMENTATION=false
opentelemetry-instrument python financial_assistant.py
```

**Benefits:**
- No code changes required
- Automatic instrumentation of all supported libraries
- Simpler deployment configuration

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- OpenAI API key OR OAuth2 LLM credentials
- (Optional) Serper API key for web search tool

### 1. Setup Environment

```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install CrewAI instrumentation
pip install -e ../[instruments]
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your credentials
```

### 3. Run the Application

**Manual Mode (default):**

```bash
# With OpenAI API key
export OPENAI_API_KEY=your-api-key
python financial_assistant.py

# With OAuth2 LLM provider
export LLM_CLIENT_ID=your-client-id
export LLM_CLIENT_SECRET=your-client-secret
export LLM_TOKEN_URL=https://your-idp/oauth2/token
export LLM_BASE_URL=https://your-llm-gateway/openai/deployments
python financial_assistant.py
```

**Zero-Code Mode:**

```bash
export OTEL_MANUAL_INSTRUMENTATION=false
opentelemetry-instrument python financial_assistant.py
```

## Environment Variables

### Instrumentation Mode

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_MANUAL_INSTRUMENTATION` | `true` for manual, `false` for zero-code | `true` |
| `OTEL_CONSOLE_OUTPUT` | Enable console span/metric output (manual mode) | `false` |

### LLM Configuration

Choose **one** of the following options:

**Option 1: OpenAI API**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |

**Option 2: OAuth2 LLM Provider**

| Variable | Description |
|----------|-------------|
| `LLM_CLIENT_ID` | OAuth2 client ID |
| `LLM_CLIENT_SECRET` | OAuth2 client secret |
| `LLM_TOKEN_URL` | OAuth2 token endpoint |
| `LLM_BASE_URL` | LLM gateway base URL |
| `LLM_APP_KEY` | (Optional) App key for request tracking |

### OpenTelemetry Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name for telemetry | `financial-trading-crew` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol | `grpc` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | GenAI emitters | `span_metric` |

## Expected Trace Structure

```
gen_ai.workflow (Financial Trading Crew)
├── gen_ai.step (Data Analysis)
│   └── invoke_agent (Data Analyst)
│       ├── chat (OpenAI/LiteLLM)
│       │   └── gen_ai.choice
│       ├── tool (Search the internet)
│       └── tool (Read website content)
├── gen_ai.step (Strategy Development)
│   └── invoke_agent (Trading Strategy Developer)
│       ├── chat (OpenAI/LiteLLM)
│       └── tool (Search the internet)
├── gen_ai.step (Execution Planning)
│   └── invoke_agent (Trade Advisor)
│       └── chat (OpenAI/LiteLLM)
└── gen_ai.step (Risk Assessment)
    └── invoke_agent (Risk Advisor)
        ├── chat (OpenAI/LiteLLM)
        └── tool (Read website content)
```

## Metrics Generated

| Metric | Description |
|--------|-------------|
| `gen_ai.workflow.duration` | Total workflow execution time |
| `gen_ai.agent.duration` | Per-agent execution time |
| `gen_ai.client.operation.duration` | LLM call latency |
| `gen_ai.client.token.usage` | Token consumption per call |

## Docker Deployment

### Build the Image

```bash
# From repository root
docker build -f instrumentation-genai/opentelemetry-instrumentation-crewai/examples/Dockerfile \
  -t financial-trading-crew:latest .
```

### Run with Docker

**Manual Mode:**

```bash
docker run --rm \
  -e OPENAI_API_KEY=your-api-key \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -e OTEL_MANUAL_INSTRUMENTATION=true \
  financial-trading-crew:latest
```

**Zero-Code Mode:**

```bash
docker run --rm \
  -e OPENAI_API_KEY=your-api-key \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -e OTEL_MANUAL_INSTRUMENTATION=false \
  financial-trading-crew:latest
```

## Kubernetes Deployment

### Create Secrets

```bash
# OpenAI credentials
kubectl create secret generic openai-credentials \
  --from-literal=api-key=your-openai-api-key \
  --from-literal=serper-api-key=your-serper-api-key

# OAuth2 LLM credentials (if using OAuth2)
kubectl create secret generic llm-credentials \
  --from-literal=client-id=your-client-id \
  --from-literal=client-secret=your-client-secret \
  --from-literal=token-url=https://your-idp/oauth2/token \
  --from-literal=base-url=https://your-llm-gateway/openai/deployments \
  --from-literal=app-key=your-app-key
```

### Deploy CronJob

```bash
# Edit cronjob.yaml to set OTEL_MANUAL_INSTRUMENTATION as needed
kubectl apply -f cronjob.yaml

# Manually trigger a job run
kubectl create job --from=cronjob/financial-trading-crew financial-trading-crew-manual-1
```

### Switch Between Modes in Kubernetes

Edit the `OTEL_MANUAL_INSTRUMENTATION` environment variable in `cronjob.yaml`:

```yaml
env:
  - name: OTEL_MANUAL_INSTRUMENTATION
    value: "false"  # Change to "false" for zero-code mode
```

## Project Structure

```
examples/
├── financial_assistant.py   # Main application (supports both modes)
├── requirements.txt         # Python dependencies
├── env.example             # Environment variable template
├── Dockerfile              # Docker build configuration
├── cronjob.yaml            # Kubernetes CronJob manifest
├── run.sh                  # Entrypoint script for Docker
└── util/
    ├── __init__.py
    └── oauth2_token_manager.py  # OAuth2 token management
```

## Troubleshooting

### No Traces Exported

1. Verify OTLP endpoint is reachable:
   ```bash
   curl -v http://localhost:4317
   ```

2. Enable console output for debugging:
   ```bash
   export OTEL_CONSOLE_OUTPUT=true
   ```

3. Check collector logs for connection issues

### OAuth2 Token Errors

1. Verify credentials are correct
2. Check token endpoint is accessible
3. Enable debug logging:
   ```bash
   export OTEL_LOG_LEVEL=debug
   ```

### Zero-Code Mode Not Working

1. Ensure `opentelemetry-distro` is installed
2. Verify `OTEL_MANUAL_INSTRUMENTATION=false`
3. Check that `opentelemetry-instrument` is in PATH

## Related Documentation

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [Splunk Observability for AI](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/set-up-observability-for-ai)
- [CrewAI Documentation](https://docs.crewai.com/)

