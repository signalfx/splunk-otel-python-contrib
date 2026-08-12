# Multi-Agent Travel Planner — Gunicorn + Azure App Service

A LangGraph multi-agent travel planner served by **Gunicorn + Uvicorn workers** with
OpenTelemetry instrumentation sending `gen_ai.*` spans and metrics to Splunk Observability Cloud.

Five specialized agents collaborate to produce a full itinerary:

```
coordinator_gc → flight_specialist_gc → hotel_specialist_gc → activity_specialist_gc → plan_synthesizer_gc
```

---

## Architecture

```
HTTP client
    │
    ▼
Azure App Service  (Gunicorn + UvicornWorker, FastAPI)
    │  OTLP/gRPC
    ▼
Azure Container Instance  (Splunk OTel Collector 0.123.0)
    │  signalfx exporter + otlphttp/splunk
    ▼
Splunk Observability Cloud  (APM traces + gen_ai.* metrics)
```

---

## Local development

### Prerequisites

```bash
pip install splunk-otel-instrumentation-langchain==0.1.14
pip install -r requirements.txt
```

Copy `.env.example` to `~/.env` and fill in your Azure OpenAI credentials and OTel settings:

```bash
cp .env.example ~/.env
# edit ~/.env
```

Start a local Splunk OTel Collector (or use `otel-tui` for quick inspection):

```bash
# Example with Docker — replace <TOKEN> and <REALM>
docker run -d --name otelcol-local \
  -p 4317:4317 -p 4318:4318 -p 13133:13133 \
  -e SPLUNK_ACCESS_TOKEN=<TOKEN> \
  -e SPLUNK_REALM=<REALM> \
  quay.io/signalfx/splunk-otel-collector:0.123.0
```

### Run the app

OTel is initialised **programmatically** inside `app.py` (see
[OTel troubleshooting — use programmatic auto-instrumentation](https://opentelemetry.io/docs/zero-code/python/troubleshooting/#use-programmatic-auto-instrumentation)).
Run Gunicorn directly — no `opentelemetry-instrument` wrapper needed:

```bash
source ~/.env

OTEL_SERVICE_NAME=multi-agent-travel-planner-gunicorn \
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=local \
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric \
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY \
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta \
OTEL_METRICS_EXPORTER=otlp \
OTEL_TRACES_EXPORTER=otlp \
OTEL_LOGS_EXPORTER=otlp \
gunicorn \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  app:app \
  --access-logfile "-" \
  --timeout 301 \
  --bind 0.0.0.0:8000
```

### Run without OTel (quick test)

```bash
source ~/.env
uvicorn app:app --port 8000
```

### Test requests

```bash
# Health check
curl http://localhost:8000/health

# Plan a trip
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"origin":"Seattle","destination":"Tokyo","travellers":2}'
```

---

## Azure deployment

### Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Contributor access to an Azure resource group
- Splunk Observability Cloud access token and realm

### Step 1 — Deploy the OTel Collector to ACI

The collector receives OTLP from the App Service and forwards to Splunk. Deploy it
first so you have the collector IP for Step 3.

```bash
export SPLUNK_ACCESS_TOKEN=<your-ingest-token>
export SPLUNK_HEC_TOKEN=<your-hec-token>
export SPLUNK_HEC_URL=https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event
export SPLUNK_REALM=<realm>          # e.g. us1
export RESOURCE_GROUP=<your-rg>
export STORAGE_ACCOUNT=<unique-lowercase-name>   # max 24 chars

chmod +x collector/deploy-aci.sh
./collector/deploy-aci.sh
```

The script creates an Azure File Share, uploads `otel-collector-config.yaml`, and
starts the container. Note the **Public IP** printed at the end.

Collector image: `quay.io/signalfx/splunk-otel-collector:0.123.0`
(see `collector/deploy-aci.sh` — override with `CONTAINER_IMAGE=...` if needed).

### Step 2 — Create the App Service plan and web app

```bash
export RESOURCE_GROUP=<your-rg>
export LOCATION=westus
export PLAN_NAME=<your-plan>
export APP_NAME=<globally-unique-app-name>

# Create Linux App Service plan (B1 is sufficient)
az appservice plan create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${PLAN_NAME}" \
  --location "${LOCATION}" \
  --is-linux \
  --sku B1

# Create the web app (Python 3.12)
az webapp create \
  --resource-group "${RESOURCE_GROUP}" \
  --plan "${PLAN_NAME}" \
  --name "${APP_NAME}" \
  --runtime "PYTHON|3.12" \
  --startup-file "sh startup.sh"

# Enable Oryx build during zip deployment.
# Without this, az webapp deployment source config-zip extracts the zip but
# does NOT run pip install, so packages from requirements.txt are missing and
# startup.sh fails with "gunicorn: not found" or import errors.
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true \
  --output none
```

### Step 3 — Configure application settings

Replace placeholders with your values. Use the collector IP from Step 1.

```bash
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings \
    AZURE_OPENAI_API_KEY="<your-api-key>" \
    AZURE_OPENAI_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/" \
    AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini" \
    AZURE_OPENAI_API_VERSION="2024-12-01-preview" \
    OTEL_EXPORTER_OTLP_ENDPOINT="http://<COLLECTOR_IP>:4317" \
    OTEL_EXPORTER_OTLP_PROTOCOL="grpc" \
    OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE="delta" \
    OTEL_SERVICE_NAME="multi-agent-travel-planner-azure" \
    OTEL_RESOURCE_ATTRIBUTES="deployment.environment=<your-env>" \
    OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric" \
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="SPAN_ONLY" \
    OTEL_METRICS_EXPORTER="otlp" \
    OTEL_TRACES_EXPORTER="otlp" \
    OTEL_LOGS_EXPORTER="otlp"
```

### Step 4 — Build and deploy the app

Run from the `gunicorn/` directory:

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/gunicorn

zip -j /tmp/gunicorn-deploy.zip app.py requirements.txt startup.sh

# config-zip triggers the Oryx build that installs requirements.txt into antenv.
# (az webapp deploy --type zip skips the Oryx build and leaves packages uninstalled.)
az webapp deployment source config-zip \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --src /tmp/gunicorn-deploy.zip
```

### Step 5 — Verify

```bash
# Health check
curl https://${APP_NAME}.azurewebsites.net/health

# Plan a trip
curl -X POST https://${APP_NAME}.azurewebsites.net/plan \
  -H "Content-Type: application/json" \
  -d '{"origin":"Seattle","destination":"Paris","travellers":2}'

# Tail live logs
az webapp log tail \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}"

# Check collector health
curl http://<COLLECTOR_IP>:13133/
```

---

## App management

```bash
# Stop (pauses billing on B1)
az webapp stop --resource-group "${RESOURCE_GROUP}" --name "${APP_NAME}"

# Start
az webapp start --resource-group "${RESOURCE_GROUP}" --name "${APP_NAME}"

# Redeploy after code changes
zip -j /tmp/gunicorn-deploy.zip app.py requirements.txt startup.sh
az webapp deployment source config-zip \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --src /tmp/gunicorn-deploy.zip

# Update collector IP if ACI was recreated
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings OTEL_EXPORTER_OTLP_ENDPOINT="http://<NEW_COLLECTOR_IP>:4317"
az webapp restart --resource-group "${RESOURCE_GROUP}" --name "${APP_NAME}"
```

---

## Expected telemetry

In **Splunk APM** (filter by `deployment.environment = <your-env>`):

| Signal | What you see |
|---|---|
| Traces | One root trace per `/plan` request; child spans per agent (`coordinator_gc`, `flight_specialist_gc`, …) |
| Spans | `gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name` on each LLM call |
| Metrics | `gen_ai.client.operation.duration` histogram, `gen_ai.client.token.usage` histogram |
| Agent view | Per-agent requests, latency, token usage, quality scores |

---

## How OTel instrumentation works

### Programmatic initialization (fork-safe)

OTel is initialized **inside the worker process** via
[`opentelemetry.instrumentation.auto_instrumentation.initialize()`](https://opentelemetry.io/docs/zero-code/python/troubleshooting/#use-programmatic-auto-instrumentation)
at the top of `app.py`, guarded so it runs exactly once per process:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

if not isinstance(trace.get_tracer_provider(), TracerProvider):
    from opentelemetry.instrumentation.auto_instrumentation import initialize
    initialize()
```

**Why not `opentelemetry-instrument gunicorn`?**

The CLI wrapper initializes the OTel SDK in the Gunicorn **master** process. After
`fork()`, only the calling thread is preserved in each worker — the
`PeriodicExportingMetricReader` timer thread is silently lost, so **metrics are never
exported** even though traces continue to flow (the `BatchSpanProcessor` is more
resilient to fork).

The programmatic approach runs `initialize()` **after** fork, giving each worker its
own fresh metric reader thread.

Reference: [Pre-fork server issues — OTel Python troubleshooting](https://opentelemetry.io/docs/zero-code/python/troubleshooting/#pre-fork-server-issues)

Support matrix for `opentelemetry-instrument` (multiple workers):

| Stack | Traces | Metrics | Logs |
|---|---|---|---|
| Uvicorn | ✓ | ✗ | ✓ |
| Gunicorn | ✓ | ✗ | ✓ |
| **Gunicorn + UvicornWorker** | **✓** | **✓** | **✓** |

> On **Linux** (Azure App Service), Gunicorn + UvicornWorker with the CLI wrapper
> also works because UvicornWorker handles fork safety. On **macOS**, gRPC C extensions
> cause a SIGSEGV after fork — use the programmatic approach for local development.

### Collector version

The ACI collector is pinned to `quay.io/signalfx/splunk-otel-collector:0.123.0`.

The `sapm` exporter was deprecated in `0.115.0` and **removed in `0.147.0`**. The
config uses `otlphttp/splunk` for traces, which is compatible with all versions from
`0.115.0` onwards.
