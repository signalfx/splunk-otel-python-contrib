# CrewAI Zero-Code Instrumentation Example

This example demonstrates **zero-code instrumentation** of CrewAI applications using `opentelemetry-instrument` with no code changes required.

## Prerequisites

1. **LLM Access** - Either OpenAI API key OR Cisco Chat AI credentials
2. **OTel Collector** (optional) - For sending telemetry to backends

## Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples/zero-code

# 1. Install dependencies
pip install -r requirements.txt

# 2. Install CrewAI instrumentation (from local source during development)
pip install -e ../../[instruments]

# 3. Configure environment variables
cp env.example .env
# Edit .env and add your credentials
```

## LLM Configuration

### Option 1: Cisco Chat AI (Default)

The example uses Cisco Chat AI via OAuth2 authentication:

```bash
# In .env
OAUTH2_CLIENT_ID=your-cisco-client-id
OAUTH2_CLIENT_SECRET=your-cisco-client-secret
OAUTH2_APP_KEY=your-cisco-app-key
```

### Option 2: OpenAI API (Direct)

To use OpenAI directly, set `OPENAI_API_KEY` and modify `customer_support.py` to remove the Cisco LLM configuration.

```bash
# In .env
OPENAI_API_KEY=your-openai-api-key
```

## OpenTelemetry Configuration (.env)

```bash
# Service name
OTEL_SERVICE_NAME=crewai-zero-code

# Local OTLP Collector
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Or Splunk Observability Cloud (HTTP required)
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://ingest.us1.signalfx.com/v2/trace/otlp
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://ingest.us1.signalfx.com/v2/datapoint/otlp
OTEL_EXPORTER_OTLP_HEADERS=X-SF-Token=YOUR_SPLUNK_TOKEN

# Enable metrics
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric

# Disable CrewAI built-in telemetry
CREWAI_DISABLE_TELEMETRY=true
```

## Run with Console Output

```bash
# Using python-dotenv to load .env file
dotenv run -- opentelemetry-instrument \
    --traces_exporter console \
    python customer_support.py
```

## Run with OTLP Exporter

```bash
# Ensure OTel collector is running on localhost:4317

dotenv run -- opentelemetry-instrument \
    --traces_exporter otlp \
    --metrics_exporter otlp \
    python customer_support.py
```

## Project Structure

```
zero-code/
├── customer_support.py    # CrewAI application with Cisco LLM integration
├── requirements.txt       # Python dependencies
├── env.example           # Environment variable template
└── README.md             # This file
```

Shared app logic lives in `../_shared/customer_support_app.py` (used by both the manual and zero-code examples).

## What Gets Instrumented

✅ **CrewAI** - Workflows, tasks, agents, tools  
✅ **OpenAI/LiteLLM** - LLM calls, token usage  
✅ **ChromaDB** - Memory queries/updates (when `memory=True`)  
✅ **HTTP** - Web scraping and external API calls

## Expected Trace Structure

```
gen_ai.workflow crew
├── gen_ai.step Task 1 (Support Inquiry)
│   └── invoke_agent Senior Support Representative
│       ├── chat gpt-4o-mini (LLM reasoning)
│       └── tool Read website content
│           └── GET https://docs.crewai.com/...
└── gen_ai.step Task 2 (QA Review)
    └── invoke_agent Support QA Specialist
        └── chat gpt-4o-mini (LLM review)
```

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OAUTH2_CLIENT_ID` | OAuth2 client ID (Cisco Chat AI in this example) | - |
| `OAUTH2_CLIENT_SECRET` | OAuth2 client secret (Cisco Chat AI in this example) | - |
| `OAUTH2_APP_KEY` | OAuth2 app key (Cisco Chat AI) | - |
| `OPENAI_API_KEY` | OpenAI API key (alternative to Cisco) | - |
| `OTEL_SERVICE_NAME` | Service name in traces | `unknown_service` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL | `http://localhost:4317` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Enable metrics (`span_metric`) | `span` |
| `CREWAI_DISABLE_TELEMETRY` | Disable CrewAI telemetry | `false` |

## Metrics Generated

When `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric`:

- `gen_ai.workflow.duration` - Total crew execution time
- `gen_ai.agent.duration` - Per-agent execution time
- `gen_ai.client.token.usage` - Token counts per LLM call
- `gen_ai.client.operation.duration` - LLM call latency

## Troubleshooting

**"Attempting to instrument while already instrumented"**  
Normal warning, safe to ignore. Means auto-instrumentation is working correctly.

**No traces appearing in console?**  
1. Verify you're using `--traces_exporter console`
2. Check that credentials are set correctly
3. Enable debug logging: `export OTEL_LOG_LEVEL=debug`

**No metrics appearing?**  
Ensure `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric` is set in your `.env` file.

**Cisco token errors?**  
1. Verify `OAUTH2_CLIENT_ID` and `OAUTH2_CLIENT_SECRET` are correct
2. Check that your credentials have access to the Chat AI API
3. Ensure `OAUTH2_APP_KEY` is set for API authorization

**OTel collector connection refused?**  
Verify your collector is running:
```bash
docker run -p 4317:4317 otel/opentelemetry-collector
```

## Production Deployment

For production, use PyPI packages:

```bash
pip install splunk-otel-instrumentation-crewai[instruments]
pip install opentelemetry-distro opentelemetry-exporter-otlp
```

Then run with:
```bash
opentelemetry-instrument python your_crewai_app.py
```
