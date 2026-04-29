---
name: run-sre-copilot
description: >-
  Complete runbook for setting up, configuring, and running the SRE Incident
  Copilot demo app with OpenTelemetry instrumentation. Use when the user asks
  to run the SRE copilot, set up the SRE demo, configure OTEL env vars for the
  copilot, use manual or zero-code instrumentation for sre_incident_copilot, or
  debug why traces/metrics are not showing up from the SRE copilot.
---

# Run SRE Incident Copilot

Multi-agent LangGraph demo that triages incidents using 4 agents + MCP tools.
Scenarios `scenario-001` through `scenario-010` are seeded in `data/alert_catalog.json`.

## 1. One-time setup

```bash
# From repo root — use the repo venv
python -m venv .venv && source .venv/bin/activate

# Core pinned deps
pip install langgraph==1.1.6 fastmcp==3.2.4

# SRE Copilot app deps
pip install -r instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot/requirements.txt

# SDOT packages in editable mode (required for local development)
pip install -e ./util/opentelemetry-util-genai
pip install -e "./instrumentation-genai/opentelemetry-instrumentation-langchain[instruments]"
pip install -e "./instrumentation-genai/opentelemetry-instrumentation-fastmcp[instruments]"

# Zero-code instrumentation support
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install
```

**Critical version pins** (newer versions break compatibility):
| Package | Pin |
|---------|-----|
| `langgraph` | `==1.1.6` |
| `fastmcp` | `==3.2.4` |

## 2. Configure environment

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot
# .env.example is added in PR #292 — copy it once merged, or create .env from the template below
cp .env.example .env   # then edit with your API key and OTLP endpoint
```

### Required `.env` fields

```dotenv
# ── LLM provider (choose one) ─────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Azure OpenAI alternative
# AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_API_VERSION=2024-02-01
# OPENAI_API_KEY=...          # reuse azure key here
# OPENAI_BASE_URL=https://your-resource.cognitiveservices.azure.com/openai/v1/
# OPENAI_MODEL=gpt-4o-mini

# ── OpenTelemetry ──────────────────────────────────────────────────────────────
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # local OTel Collector (gRPC)
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_SERVICE_NAME=sre-incident-copilot
OTEL_TRACES_EXPORTER=otlp_proto_grpc                # required for zero-code
OTEL_METRICS_EXPORTER=otlp_proto_grpc               # required for zero-code
OTEL_METRIC_EXPORT_INTERVAL=5000                    # 5 s flush for demos
OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric     # traces + metrics

# ── MCP subprocess instrumentation ────────────────────────────────────────────
SRE_COPILOT_MCP_USE_OTEL_WRAPPER=true               # wrap MCP servers with otel-instrument
```

> Shell env always wins over `.env` — `load_dotenv(override=False)` is in `main.py`.

## 3. Run commands

All commands run from the `sre_incident_copilot/` directory.

### Manual instrumentation (SDK configured in `main.py`)

```bash
python main.py --scenario scenario-001 --manual-instrumentation
```

With full demo flags (interrupt/resume, metric flush wait):

```bash
python main.py \
  --scenario scenario-001 \
  --simulate-interrupt-resume \
  --conversation-id troubleshooting-chat-1 \
  --manual-instrumentation \
  --wait-after-completion 15
```

As Workflow root span instead of default AgentInvocation:

```bash
python main.py --scenario scenario-001 --manual-instrumentation \
  --root-as-workflow "SRE Incident Copilot"
```

### Zero-code instrumentation (`opentelemetry-instrument` wrapper)

```bash
opentelemetry-instrument python main.py --scenario scenario-001
```

> **Note**: `OTEL_TRACES_EXPORTER=otlp_proto_grpc` and `OTEL_METRICS_EXPORTER=otlp_proto_grpc`
> must be set — the generic `otlp` entry point is often missing from venvs.

### Cross-process interrupt / resume

```bash
CONV="troubleshooting-chat-$(date +%s)"

# 1. Run to interrupt point
python main.py --scenario scenario-001 --enable-interrupt --conversation-id $CONV

# 2. Resume (approve)
python main.py --scenario scenario-001 --resume --approve --conversation-id $CONV

# 2b. Resume (reject with feedback)
python main.py --scenario scenario-001 --resume --reject \
  --feedback "Need more evidence" --conversation-id $CONV
```

## 4. Available scenarios

`scenario-001` through `scenario-010` — all defined in `data/alert_catalog.json`.

Typical quick demo: `scenario-001` (payment-service latency spike).

## 5. Local OTel Collector (optional)

Start the collector that forwards to Splunk O11y Cloud:

```bash
# From repo root
docker compose -f .local/otelcol-docker-compose.yaml up -d
```

Requires `.local/otelcol-config.yaml` with `SPLUNK_ACCESS_TOKEN` and `SPLUNK_REALM`.
For console-only output (no collector), replace the OTEL endpoint with the console exporter:

```bash
OTEL_TRACES_EXPORTER=console python main.py --scenario scenario-001 --manual-instrumentation
```

## 6. Key CLI flags reference

| Flag | Description |
|------|-------------|
| `--scenario` | Scenario ID, e.g. `scenario-001` |
| `--manual-instrumentation` | Use SDK configured in `main.py` instead of zero-code |
| `--conversation-id` | Pin `gen_ai.conversation.id` across runs |
| `--simulate-interrupt-resume` | Two traces, one conversation (single process) |
| `--enable-interrupt` | Pause at action planner for human review |
| `--resume --approve` | Resume and approve the mitigation plan |
| `--resume --reject --feedback "..."` | Resume and reject with feedback |
| `--root-as-workflow NAME` | Root span as Workflow instead of AgentInvocation |
| `--wait-after-completion N` | Sleep N seconds for telemetry flush |

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| MCP server spans missing | Check `SRE_COPILOT_MCP_USE_OTEL_WRAPPER=true` and `opentelemetry-instrument` is on `PATH` |
| `otlp` exporter not found | Set `OTEL_TRACES_EXPORTER=otlp_proto_grpc` and `OTEL_METRICS_EXPORTER=otlp_proto_grpc` |
| Metrics not appearing | Check `OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric`; add `--wait-after-completion 15` |
| `langgraph` compatibility errors | Downgrade: `pip install langgraph==1.1.6` |
| `fastmcp` import errors | Downgrade: `pip install fastmcp==3.2.4` |
| `OPENAI_API_KEY` not found | Ensure `.env` is populated or export the key in shell |
