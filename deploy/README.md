# deploy/

Local development deployment helpers for the Splunk OpenTelemetry Python Contrib repo.

## Contents

| File | Purpose |
|------|---------|
| `otelcol-docker-compose.yaml` | Docker Compose to run a local OTel Collector container |
| `otelcol-config.yaml` | Collector config: receives OTLP, forwards traces/logs to Splunk O11y Cloud (OTLP HTTP) and metrics via SignalFx exporter |
| `run-sre-copilot-skill.md` | Cursor agent skill — runbook for setting up and running the SRE Incident Copilot demo |

---

## Local OTel Collector

Starts an `otel/opentelemetry-collector-contrib` container listening on:
- `localhost:4317` — OTLP gRPC
- `localhost:4318` — OTLP HTTP
- `localhost:13133` — health check

### Prerequisites

```bash
export SPLUNK_ACCESS_TOKEN="<your-ingest-token>"
export SPLUNK_REALM="us1"   # or us0, eu0, etc.
```

### Start

```bash
docker compose -f deploy/otelcol-docker-compose.yaml up -d
```

### Stop

```bash
docker compose -f deploy/otelcol-docker-compose.yaml down
```

### App env vars

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
```

---

---

## dev-assistant as a Cursor MCP Server

`dev_assistant_server.py` exposes 7 tools (list_files, read_file, write_file, run_command, git_status, search_code, get_system_info) that Cursor can use as an MCP server.

The workspace `.cursor/mcp.json` already contains a ready-to-use entry:

```json
{
  "mcpServers": {
    "dev-assistant": {
      "command": "<repo>/.venv/bin/python",
      "args": ["<repo>/instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/dev_assistant_server.py"],
      "env": {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_SERVICE_NAME": "mcp-dev-assistant-server",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": "span_metric",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"
      }
    }
  }
}
```

**To activate**: open Cursor Settings → MCP → reload, or restart Cursor. The `dev-assistant` server will appear in the MCP panel.

**Prerequisites for server spans**:

```bash
# Create .env in the examples folder (inherited by the server subprocess)
cp instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/.env.example \
   instrumentation-genai/opentelemetry-instrumentation-fastmcp/examples/.env

# Start the local OTel Collector
docker compose -f deploy/otelcol-docker-compose.yaml up -d
```

---

## SRE Incident Copilot Skill

`run-sre-copilot-skill.md` is a [Cursor agent skill](https://docs.cursor.com/context/rules) that teaches the AI agent how to set up and run the SRE Incident Copilot demo without re-explaining the setup each session.

**Install the skill** (personal, available across all projects):

```bash
mkdir -p ~/.cursor/skills/run-sre-copilot
cp deploy/run-sre-copilot-skill.md ~/.cursor/skills/run-sre-copilot/SKILL.md
```

**Trigger phrases**: "run the SRE copilot", "set up SRE demo", "configure OTEL for copilot"

See the skill file for the full runbook (env vars, versions, commands, troubleshooting).
