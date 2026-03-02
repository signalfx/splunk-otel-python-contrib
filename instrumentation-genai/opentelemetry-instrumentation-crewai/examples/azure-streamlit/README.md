# OpenTelemetry for Streamlit + CrewAI + Azure App Service

This example deploys a **Streamlit** UI for a **CrewAI** customer-support agent that uses **Azure OpenAI** as its LLM and embedding provider. Full end-to-end OpenTelemetry traces are sent to **Splunk Observability Cloud** via an OTel Collector running in **Azure Container Instance** (gateway mode).

## Architecture

```
Browser
  │  HTTPS
  ▼
Azure App Service  (Streamlit + Tornado, Python 3.12)
  │  OTel initialised once via sys.modules guard in app.py
  │  OTLP/gRPC  port 4317
  ▼
Azure Container Instance — Splunk OTel Collector (gateway mode)
  │  SAPM / SignalFx / HEC
  ▼
Splunk Observability Cloud (APM traces + metrics + logs)
```

> **Streamlit runs under its own Tornado web server**, not Gunicorn. The custom `startup.sh` startup command bypasses App Service's default Gunicorn wrapper and calls `streamlit run` directly. See [Known issues](#known-issues-and-workarounds) for what this means for OTel initialisation.

## Prerequisites

- Python ≥ 3.10 (Python 3.12 used in this example)
- Azure CLI installed and logged in (`az login`)
- Azure OpenAI resource with:
  - A chat completion deployment (e.g. `gpt-4.1-nano`)
  - A text embedding deployment (e.g. `text-embedding-3-large`)
- A Splunk Observability Cloud access token with ingest permissions

## Project structure

```
azure-streamlit/
├── app.py                      # Streamlit UI + programmatic OTel init
├── customer_support_azure.py   # CrewAI agents, tasks, crew definition
├── requirements.txt            # Pinned Python dependencies
├── run.sh                      # Local development launcher
├── startup.sh                  # Azure App Service startup command
├── .env.example                # Template for local environment variables
└── collector/
    ├── otel-collector-config.yaml  # Splunk OTel Collector gateway config
    ├── deploy-aci.sh               # One-shot ACI deployment script
    └── README.md                   # Collector deployment guide
```

---

## Local development

### 1. Clone and set up environment

```bash
git clone https://github.com/signalfx/splunk-otel-python-contrib
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples/azure-streamlit

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `pysqlite3-binary` is listed in `requirements.txt`. It bundles a modern SQLite ≥ 3.35.0 and is monkey-patched in `app.py` before any `crewai` import. This is required because `chromadb` (a `crewai-tools` dependency) requires SQLite ≥ 3.35.0, which many Linux distributions (including Azure App Service) do not ship by default.

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

| Variable | Description | Required |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT` | Chat completion deployment name | Yes |
| `AZURE_OPENAI_API_VERSION` | Chat completion API version | Yes |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name | Yes |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | Embedding API version | Yes |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTel Collector gRPC endpoint | Yes |
| `OTEL_SERVICE_NAME` | Service name shown in Splunk APM | Yes |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture LLM prompts/responses | Optional |
| `CREWAI_DISABLE_TELEMETRY` | Disable CrewAI's own built-in telemetry | Recommended |

### 3. Run locally

```bash
bash run.sh
```

The app opens at `http://localhost:8501`. OTel is initialised programmatically inside `app.py`; no `opentelemetry-instrument` wrapper is needed.

---

## Deployment

### App Service — deploy the Streamlit app

#### 1. Create the App Service (first time only)

```bash
RESOURCE_GROUP="o11y-inframon-ai"
APP_NAME="customersupportcrewai"
PLAN_NAME="crewai-plan"
LOCATION="westus"

# Create App Service Plan (B2 or higher recommended for CrewAI memory)
az appservice plan create \
  --name "${PLAN_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --is-linux \
  --sku B2

# Create the Web App
az webapp create \
  --resource-group "${RESOURCE_GROUP}" \
  --plan "${PLAN_NAME}" \
  --name "${APP_NAME}" \
  --runtime "PYTHON|3.12" \
  --startup-file "sh startup.sh"
```

#### 2. Configure application settings

```bash
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings \
    AZURE_OPENAI_API_KEY="<your-api-key>" \
    AZURE_OPENAI_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/" \
    AZURE_OPENAI_DEPLOYMENT="gpt-4.1-nano" \
    AZURE_OPENAI_API_VERSION="2024-12-01-preview" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large" \
    AZURE_OPENAI_EMBEDDING_API_VERSION="2024-02-01" \
    OTEL_EXPORTER_OTLP_ENDPOINT="http://<collector-ip>:4317" \
    OTEL_SERVICE_NAME="crewai-customer-support-ui" \
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true" \
    OTEL_RESOURCE_ATTRIBUTES="deployment.environment=<your-env>" \
    CREWAI_DISABLE_TELEMETRY="true"
```

#### 3. Package and deploy

```bash
cd instrumentation-genai/opentelemetry-instrumentation-crewai/examples

zip azure-streamlit-deploy.zip \
  azure-streamlit/app.py \
  azure-streamlit/customer_support_azure.py \
  azure-streamlit/requirements.txt \
  azure-streamlit/startup.sh

# Repackage with files at the zip root (Oryx requirement)
cd azure-streamlit
zip ../azure-streamlit-deploy.zip \
  app.py customer_support_azure.py requirements.txt startup.sh

az webapp deploy \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --src-path azure-streamlit-deploy.zip \
  --type zip
```

#### 4. Verify the deployment

```bash
# Check app status
az webapp show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --query "{state:state,url:defaultHostName}" \
  --output table

# Tail live logs
az webapp log tail \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}"
```

App URL: `https://<APP_NAME>.azurewebsites.net`

#### Stop the app (pause billing)

```bash
az webapp stop --resource-group "${RESOURCE_GROUP}" --name "${APP_NAME}"
```

#### Start the app

```bash
az webapp start --resource-group "${RESOURCE_GROUP}" --name "${APP_NAME}"
```

#### Reset credentials before leaving (avoid misuse)

Set sensitive settings to placeholder values when not in active use:

```bash
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings \
    AZURE_OPENAI_API_KEY="Not set" \
    AZURE_OPENAI_ENDPOINT="Not set" \
    OTEL_EXPORTER_OTLP_ENDPOINT="Not set"
```

---

### OTel Collector — deploy to Azure Container Instance

See [collector/README.md](collector/README.md) for the full guide.

**Required environment variables** — set these before running the deploy script:

| Variable | Description | Example |
|---|---|---|
| `SPLUNK_ACCESS_TOKEN` | Splunk Observability Cloud ingest token | `gXgmP9v-...` |
| `SPLUNK_HEC_TOKEN` | Splunk HEC token for log ingestion | `bdef2e63-...` |
| `SPLUNK_HEC_URL` | Splunk HEC endpoint URL | `https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event` |
| `SPLUNK_REALM` | Splunk Observability Cloud realm | `us1` |
| `RESOURCE_GROUP` | Azure resource group | `my-resource-group` |
| `STORAGE_ACCOUNT` | Storage account name (globally unique, lowercase, max 24 chars) | `myotelcfgstorage` |

**Optional overrides** (defaults shown):

| Variable | Default | Description |
|---|---|---|
| `LOCATION` | `westus` | Azure region (must match the resource group) |
| `CONTAINER_NAME` | `splunk-otel-collector` | ACI container name |
| `DEPLOYMENT_ENV` | `azure` | Value for the `deployment.environment` resource attribute |
| `SPLUNK_MEMORY_LIMIT_MIB` | `900` | Memory ceiling for the collector's memory_limiter processor |

**Quick deploy:**

```bash
export SPLUNK_ACCESS_TOKEN=<your-token>
export SPLUNK_HEC_TOKEN=<your-hec-token>
export SPLUNK_HEC_URL=https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event
export SPLUNK_REALM=us1
export RESOURCE_GROUP=<your-resource-group>
export STORAGE_ACCOUNT=<your-storage-account>

cd collector
chmod +x deploy-aci.sh
./deploy-aci.sh
```

After deployment, set `OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-ip>:4317` in the App Service settings above.

---

## Trace hierarchy

The instrumentation creates a trace hierarchy similar to the following:

```
crewai.crew.kickoff  (manual root span — anchors all agent spans to one trace)
└── Crew: customer_support_crew  (Workflow)
    ├── Task: inquiry_resolution  (Step)
    │   └── Agent: Senior Support Representative  (AgentInvocation)
    │       ├── embeddings text-embedding-3-large  (memory lookup)
    │       ├── chat gpt-4.1-nano  (LLM call via openai-instrumentation)
    │       └── Tool: docs_scrape  (ToolCall)
    └── Task: quality_assurance  (Step)
        └── Agent: Support Quality Assurance Specialist  (AgentInvocation)
            ├── embeddings text-embedding-3-large  (memory lookup)
            └── chat gpt-4.1-nano  (LLM call via openai-instrumentation)
```

Each span includes attributes such as:

- `gen_ai.system` = `crewai`
- `gen_ai.operation.name` = `invoke_workflow` | `invoke_agent` | `execute_tool`
- `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- Agent role, task description, tool names

---

## Known issues and workarounds

### sqlite3 version on Azure App Service

Azure App Service (Debian Bullseye) ships `sqlite3 < 3.35.0`. `chromadb` (a `crewai-tools` dependency for memory storage) requires `sqlite3 >= 3.35.0` and raises `RuntimeError` at import time.

**Fix:** `pysqlite3-binary` is included in `requirements.txt` and monkey-patched at the very top of `app.py` before any `crewai` import:

```python
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
```

### OTel instrumentation must be initialised after the sqlite3 patch

Using `opentelemetry-instrument` as a process wrapper (the typical zero-code approach) initialises instrumentors *before* `app.py` runs. The CrewAI instrumentor imports `crewai` at that point, which triggers the `chromadb` sqlite3 check and fails silently. This leaves CrewAI spans without a parent, producing split traces.

**Fix:** OTel is initialised programmatically *inside* `app.py`, after the sqlite3 patch:

```python
_OTEL_INIT_KEY = "__crewai_otel_initialized__"
if _OTEL_INIT_KEY not in sys.modules:
    sys.modules[_OTEL_INIT_KEY] = True
    from opentelemetry.instrumentation.auto_instrumentation import initialize
    initialize()
```

### Streamlit re-execution causes duplicate spans

This is the most subtle issue and has nothing to do with Gunicorn or Tornado workers. Because we use a custom `startup.sh` startup command, App Service bypasses Gunicorn entirely and Streamlit runs under its own **Tornado** web server. There are no multiple workers or forked processes involved.

The real cause is **Streamlit's reactive execution model**: Streamlit re-runs `app.py` top-to-bottom on *every user interaction* (button click, form submit, widget change) in the same Python process using `exec()`. Without a guard, `initialize()` is called on every rerun and each call wraps CrewAI and OpenAI functions again — on top of the existing patches. After two submits the functions are double-wrapped; after three, triple-wrapped — producing N× duplicate spans per crew run. This is exactly what caused the 3× `workflow`, 3× `invoke_agent` spans observed in Splunk APM.

```
User submits form
      │
      ▼
Streamlit reruns app.py via exec() in the same process
      │
      ├── initialize()  ← called again, wraps crewai functions a second time
      │
      └── crew.kickoff() → 2× workflow, 2× invoke_agent, 2× embeddings ...
```

**Why module-level variables don't work:** Streamlit resets `module.__dict__` on each rerun, so a top-level assignment like `_INITIALIZED = False` is reassigned to `False` every time, making it useless as a guard.

**Fix:** `sys.modules` is Python's global module registry — a process-level dict that lives outside the script's namespace and is never touched by Streamlit's `exec()`. A sentinel key written there survives all reruns for the lifetime of the process:

```python
_OTEL_INIT_KEY = "__crewai_otel_initialized__"
if _OTEL_INIT_KEY not in sys.modules:   # True only on the very first run
    sys.modules[_OTEL_INIT_KEY] = True  # persists across all subsequent reruns
    from opentelemetry.instrumentation.auto_instrumentation import initialize
    initialize()                        # patches applied exactly once
```

This is the correct pattern for any process-level one-time setup (OTel init, DB connection pools, etc.) in a Streamlit application.

---

## References

- [CrewAI documentation](https://docs.crewai.com/)
- [OpenTelemetry Python documentation](https://opentelemetry.io/docs/languages/python/)
- [OpenTelemetry zero-code Python troubleshooting](https://opentelemetry.io/docs/zero-code/python/troubleshooting/)
- [Azure App Service Python configuration](https://learn.microsoft.com/en-us/azure/app-service/configure-language-python)
- [Splunk GenAI utilities](https://github.com/signalfx/splunk-otel-python-contrib)
- [Splunk OTel Collector](https://docs.splunk.com/observability/en/gdi/opentelemetry/opentelemetry.html)

## License

Apache-2.0
