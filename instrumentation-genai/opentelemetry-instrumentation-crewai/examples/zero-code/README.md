# CrewAI Zero-Code Instrumentation Example

This example demonstrates **zero-code instrumentation** of CrewAI applications using `opentelemetry-instrument` with no code changes required.

## Prerequisites

1. **OpenAI API Key** - Required for LLM calls
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
# Edit .env and add your OPENAI_API_KEY
```

## Configuration (.env)

Create a `.env` file with:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=your-openai-api-key-here

# OpenTelemetry Configuration
OTEL_SERVICE_NAME=crewai-zero-code
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Enable metrics (required for gen_ai.agent.duration, gen_ai.workflow.duration)
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric

# Disable CrewAI built-in telemetry (recommended)
CREWAI_DISABLE_TELEMETRY=true

# OpenAI Model
OPENAI_MODEL_NAME=gpt-4o-mini
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

## What Gets Instrumented

✅ **CrewAI** - Workflows, tasks, agents, tools  
✅ **OpenAI** - LLM calls, token usage, embeddings  
✅ **ChromaDB** - Memory queries/updates (when `memory=True`)  
✅ **HTTP** - Web scraping and external API calls

## Expected Trace Structure

```
gen_ai.workflow crew
├── gen_ai.step Task 1 (Support Inquiry)
│   └── invoke_agent Senior Support Representative
│       ├── chroma.query (memory retrieval)
│       ├── embeddings text-embedding-3-small
│       ├── chat gpt-4o-mini (LLM reasoning)
│       └── tool Read website content
│           └── GET https://docs.crewai.com/...
└── gen_ai.step Task 2 (QA Review)
    └── invoke_agent Support QA Specialist
        ├── chroma.query (memory retrieval)
        ├── embeddings text-embedding-3-small
        └── chat gpt-4o-mini (LLM review)
```

## Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (**required**) | - |
| `OTEL_SERVICE_NAME` | Service name in traces | `unknown_service` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL | `http://localhost:4317` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Enable metrics (`span_metric`) | `span` |
| `CREWAI_DISABLE_TELEMETRY` | Disable CrewAI telemetry | `false` |
| `OPENAI_MODEL_NAME` | Default OpenAI model | `gpt-4o-mini` |

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
2. Check that `OPENAI_API_KEY` is set correctly
3. Enable debug logging: `export OTEL_LOG_LEVEL=debug`

**No metrics appearing?**  
Ensure `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric` is set in your `.env` file.

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
