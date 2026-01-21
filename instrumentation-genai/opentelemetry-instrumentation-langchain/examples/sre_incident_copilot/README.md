# SRE Incident Copilot

A production-style demo application showcasing GenAI observability capabilities using LangChain/LangGraph with multi-agent orchestration, MCP tools, and evaluation metrics.

## Overview

The SRE Incident Copilot demonstrates an agentic workflow that:

- Investigates alerts using realistic observability tooling (metrics/logs/traces search via MCP)
- Produces actionable incident artifacts (tasks, notifications, postmortem drafts)
- Emits comprehensive OpenTelemetry telemetry for the agent workflow
- Detects model/tool drift with evals and enforces safety gates

## Architecture

### Agents

1. **Triage Agent**: Normalizes alerts, identifies affected services, selects investigation plan
2. **Investigation Agent**: Queries metrics/logs/traces, assembles evidence, proposes hypotheses
3. **Action Planner Agent**: Translates hypotheses into mitigation steps and tasks
4. **Quality Gate Agent**: Enforces safety rails, validates outputs, computes eval metrics

### Tools

- **MCP Tools**: `metrics_query`, `logs_search`, `trace_query` (via MCP server)
- **RAG Tools**: `runbook_search` (vector search over runbooks with citations)
- **Integration Tools**: `service_catalog_lookup`, `task_writer`, `notifier`
- **Agent-as-Tool**: `investigation_agent_mcp` (Investigation Agent exposed as MCP tool)

### Workflow

```mermaid
graph LR
    A[Alert] --> B[Triage Agent]
    B --> C[Investigation Agent]
    C --> D[Action Planner Agent]
    D --> E[Quality Gate Agent]
    E --> F[Artifacts]

    C --> G[MCP Tools]
    B --> H[Runbook RAG]
    D --> I[Task Writer]
```

## Features

- **Multi-Agent Orchestration**: 4 agents working in sequence with LangGraph
- **MCP Tools**: Observability tools exposed via MCP protocol
- **Agent-as-MCP-Tool**: Investigation Agent can be called as an MCP tool
- **RAG with Citations**: Runbook search with vector embeddings and citations
- **Seeded Data**: Deterministic scenarios with consistent telemetry
- **Evaluation Metrics**: DeepEval integration for quality assessment
- **Drift Simulation**: Simulation runner with configurable drift modes
- **K8s Ready**: CronJob configuration for scheduled runs

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- OpenTelemetry collector endpoint (optional)

### Installation

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Important**: Always activate the virtual environment before running the application:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Configuration

The application supports two separate credential configurations for LLM and embeddings:

#### Option 1: OpenAI (both chat and embeddings)

Set the following environment variables:

```bash
export OPENAI_API_KEY=sk-your-openai-api-key
export OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

#### Option 2: Circuit + Azure (Circuit for chat, Azure for embeddings)

Set the following environment variables:

```bash
# Circuit/Cisco credentials (all 5 required)
export CIRCUIT_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini
export CIRCUIT_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
export CIRCUIT_CLIENT_ID=your-client-id
export CIRCUIT_CLIENT_SECRET=your-client-secret
export CIRCUIT_APP_KEY=your-app-key

# Azure OpenAI credentials (all 4 required)
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-azure-api-key
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

**Note:** Circuit requires **all 5** variables to be set. Azure requires **all 4** variables to be set. If any are missing, the application will fall back to OpenAI credentials.

#### Additional Configuration

```bash
# OpenTelemetry (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=sre-incident-copilot

# Application settings
export SCENARIO_ID=scenario-001
export DATA_DIR=data
export ARTIFACTS_DIR=artifacts
```

You can also create a `.env` file with these variables for convenience.

## Running the Application

### Prerequisites Checklist

Before running the application, ensure:

1. ✅ Virtual environment is activated
2. ✅ Dependencies are installed (`pip install -r requirements.txt`)
3. ✅ Credentials are configured (either OpenAI OR Circuit+Azure)

### Step-by-Step Guide

**1. Activate the virtual environment:**

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**2. Set up credentials:**

Choose **one** of the following options:

**Option A - OpenAI:**
```bash
export OPENAI_API_KEY=sk-your-openai-api-key
```

**Option B - Circuit + Azure:**
```bash
# All Circuit variables required
export CIRCUIT_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini
export CIRCUIT_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
export CIRCUIT_CLIENT_ID=your-client-id
export CIRCUIT_CLIENT_SECRET=your-client-secret
export CIRCUIT_APP_KEY=your-app-key

# All Azure variables required
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-azure-api-key
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

**3. Run a scenario:**

```bash
python main.py --scenario scenario-001
```

### Troubleshooting

**Error: "No valid credentials configured"**
- Ensure you've exported all required environment variables
- For Circuit: all 5 variables must be set
- For Azure: all 4 variables must be set
- For OpenAI: `OPENAI_API_KEY` must be set

**Error: "404 Not Found"**
- Check that `CIRCUIT_BASE_URL` does NOT include `/chat/completions` suffix
- Correct: `https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini`
- Incorrect: `https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini/chat/completions`

## Usage

### Quick Start Examples

**Run with OpenAI:**
```bash
export OPENAI_API_KEY=sk-your-key
python main.py --scenario scenario-001
```

**Run with Circuit + Azure:**
```bash
export CIRCUIT_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini
export CIRCUIT_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
export CIRCUIT_CLIENT_ID=your-id
export CIRCUIT_CLIENT_SECRET=your-secret
export CIRCUIT_APP_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-azure-key
export AZURE_OPENAI_API_VERSION=2024-02-01
export AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
python main.py --scenario scenario-001
```

### Run a Single Scenario

**Make sure the venv is activated first:**

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python main.py --scenario scenario-001
```

### Run with Manual Instrumentation

```bash
python main.py --scenario scenario-001 --manual-instrumentation
```

### Run Simulation with Drift

**Make sure the venv is activated first:**

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python simulation_runner.py \
  --scenarios scenario-001 scenario-002 \
  --iterations 10 \
  --drift-mode tool_failure \
  --drift-intensity 0.2
```

### Validation

Validation is performed automatically when running scenarios. The `main.py` script runs business logic validation and displays results. To validate a scenario:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python main.py --scenario scenario-001
```

Validation checks:

- Hypothesis matching (top hypothesis contains expected root cause keywords)
- Evidence sufficiency (collects expected evidence types)
- Action safety (actions are safe given confidence level)

## Available Scenarios

- `scenario-001`: Database connection pool exhaustion
- `scenario-002`: Cache miss storm
- `scenario-003`: Recent deployment issue
- `scenario-004`: Database connection pool exhaustion (different service)
- `scenario-005`: Redis cache memory pressure
- `scenario-006`: Authentication service failures
- `scenario-007`: Notification queue depth runaway
- `scenario-008`: Analytics service latency degradation
- `scenario-009`: Payment service deployment correlation
- `scenario-010`: User service dependency failure

## Outputs

Each run produces artifacts in `artifacts/<run_id>/`:

- `inputs.json`: Input alert and scenario details
- `outputs.json`: Agent outputs and results
- `run_meta.json`: Run metadata and configuration
- `incident_summary.md`: Formatted incident summary
- `postmortem_draft.md`: Postmortem template
- `eval_report.json`: Evaluation metrics and scores

## Kubernetes Deployment

See `k8s-cronjob.yaml` for a complete CronJob configuration. The app runs every 10 minutes by default.

```bash
kubectl apply -f k8s-cronjob.yaml
```

## Development

## Key Files

| File                                   | Purpose                                                    |
| -------------------------------------- | ---------------------------------------------------------- |
| `main.py`                              | Workflow orchestration, artifact generation                |
| `agents.py`                            | Triage, Investigation, Action Planner, Quality Gate agents |
| `tools.py`                             | LangChain tools including MCP tool wrappers                |
| `mcp_tools/observability_tools.py`     | MCP server for metrics/logs/traces                         |
| `mcp_tools/investigation_agent_mcp.py` | Investigation Agent as MCP tool                            |
| `runbook_search.py`                    | RAG implementation for runbook search                      |
| `validation.py`                        | Business logic validation harness                          |
| `simulation_runner.py`                 | Batch simulation with drift modes                          |

## Environment Variables

| Variable                      | Purpose                        | Default                |
| ----------------------------- | ------------------------------ | ---------------------- |
| `OPENAI_API_KEY`              | OpenAI API key                 | Optional*              |
| `OPENAI_MODEL`                | Model to use                   | `gpt-4o-mini`          |
| `OPENAI_BASE_URL`             | OpenAI custom endpoint         | Optional               |
| `CIRCUIT_BASE_URL`            | Circuit/Cisco endpoint         | Optional               |
| `CIRCUIT_TOKEN_URL`           | Circuit OAuth2 token endpoint  | Optional*              |
| `CIRCUIT_CLIENT_ID`           | Circuit OAuth2 client ID       | Optional*              |
| `CIRCUIT_CLIENT_SECRET`       | Circuit OAuth2 client secret   | Optional*              |
| `CIRCUIT_APP_KEY`             | Circuit OAuth2 app key         | Optional               |
| `AZURE_OPENAI_ENDPOINT`       | Azure OpenAI endpoint          | Optional               |
| `AZURE_OPENAI_API_KEY`        | Azure OpenAI API key           | Optional               |
| `AZURE_OPENAI_API_VERSION`    | Azure OpenAI API version       | `2024-02-01`           |
| `AZURE_EMBEDDING_DEPLOYMENT`  | Azure embedding deployment     | Optional               |
| `OTEL_SERVICE_NAME`           | Service name for telemetry     | `sre-incident-copilot` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint                  | Optional               |
| `SCENARIO_ID`                 | Scenario to run                | Required               |
| `DATA_DIR`                    | Data directory                 | `data`                 |
| `ARTIFACTS_DIR`               | Artifacts directory            | `artifacts`            |
| `CONFIDENCE_THRESHOLD`        | Minimum confidence for actions | `0.7`                  |
| `EVIDENCE_COUNT_THRESHOLD`    | Minimum evidence pieces        | `3`                    |
