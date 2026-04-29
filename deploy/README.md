# deploy/

Local development deployment helpers for the Splunk OpenTelemetry Python Contrib repo.

## Contents

| File | Purpose |
|------|---------|
| `otelcol-docker-compose.yaml` | Docker Compose — runs Splunk Distro OTel Collector on `localhost:4317/4318` |
| `otelcol-config.yaml` | Collector config: OTLP receiver → traces/logs via `otlphttp/splunk`, metrics via `signalfx` (with `send_otlp_histograms: true`) |
| `.env.example` | Template for `SPLUNK_ACCESS_TOKEN` and `SPLUNK_REALM` — copy to `.env` before starting |
| `run-sre-copilot-skill.md` | Cursor agent skill — runbook for setting up and running the SRE Incident Copilot demo |

---

## Local OTel Collector

Uses the **Splunk Distribution of the OTel Collector** (`quay.io/signalfx/splunk-otel-collector`) — required for `send_otlp_histograms: true` in the signalfx exporter. The upstream `otel/opentelemetry-collector-contrib` image does not support that flag.

Listens on:
- `localhost:4317` — OTLP gRPC
- `localhost:4318` — OTLP HTTP
- `localhost:13133` — health check

### Prerequisites

```bash
# Copy and fill in your Splunk O11y credentials
cp deploy/.env.example deploy/.env
# Edit deploy/.env: set SPLUNK_ACCESS_TOKEN and SPLUNK_REALM
```

The Splunk Distro validates `access_token` at **startup** (not just at export time) — the container will exit immediately if the token is empty.

### Start

```bash
docker compose -f deploy/otelcol-docker-compose.yaml up -d
# Verify
curl http://localhost:13133
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

## SRE Incident Copilot Skill

`run-sre-copilot-skill.md` is a [Cursor agent skill](https://docs.cursor.com/context/rules) that teaches the AI agent how to set up and run the SRE Incident Copilot demo without re-explaining the setup each session.

**Install the skill** (personal, available across all projects):

```bash
mkdir -p ~/.cursor/skills/run-sre-copilot
cp deploy/run-sre-copilot-skill.md ~/.cursor/skills/run-sre-copilot/SKILL.md
```

**Trigger phrases**: "run the SRE copilot", "set up SRE demo", "configure OTEL for copilot"

See the skill file for the full runbook (env vars, versions, commands, troubleshooting).
