# Multi-Agent Travel Planner (LangGraph + LangChain + OpenTelemetry)

This example shows a small team of LangChain agents connected by a LangGraph state machine to produce a week-long city break itinerary (flights, hotel, activities, synthesis). It demonstrates GenAI observability (traces, metrics, logs, evaluation metrics) using the OpenTelemetry LangChain/LangGraph instrumentation.

## 1. Architecture Overview

Agents (ReAct-style):

1. `coordinator` – Interprets traveller request and outlines a plan.
2. `flight_specialist` – Suggests flight option (tool: `mock_search_flights`).
3. `hotel_specialist` – Recommends hotel (tool: `mock_search_hotels`).
4. `activity_specialist` – Curates activities (tool: `mock_search_activities`).
5. `plan_synthesizer` – Produces final structured itinerary.

Note: for LangGraph nodes to be recognized as `AgentInvocation` by the instrumentation, it has to have the following configuration

- `agent_name` metadata in the LangGraph config, i.e. 

```python
_create_react_agent(llm, tools=[]).with_config(
    {
        "metadata": {
            "agent_name": "coordinator",
        },
    }
```

- `agent` tags in the config, i.e 

```python
_create_react_agent(llm, tools=[]).with_config(
    {
        "tags": ["agent:coordinator"],
    }
```

See more example in `main.py` example.

LangGraph `StateGraph` drives transitions; `should_continue` moves through sequence until `END`. State (`PlannerState`) accumulates messages, per-agent summaries, poison events, and the final itinerary.

```text
[User Request] --> Pre-Parse (origin/dest/dates) --> START
							 |
							 v
					 LangGraph Workflow
	    +-------------+----------+-------------+---------------+
	    |             |          |             |               |
 [Coordinator] -> [Flight] -> [Hotel] -> [Activities] -> [Synthesizer] -> END
	    |             |          |             |               |
	    +-------------+----------+-------------+---------------+
```

## 2. Prompt Poisoning (Quality Noise)

To exercise evaluation dimensions, the app can inject mild quality-degrading snippets into agent prompts (hallucination, bias, irrelevance, negative sentiment, toxicity). Controlled by env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRAVEL_POISON_PROB` | `0.8` | Probability a step is poisoned |
| `TRAVEL_POISON_TYPES` | `hallucination,bias,irrelevance,negative_sentiment,toxicity` | Allowed poison types |
| `TRAVEL_POISON_MAX` | `2` | Max snippets injected per step |
| `TRAVEL_POISON_SEED` | (unset) | Deterministic seed |

Injected kinds recorded in `state['poison_events']` and can surface evaluation metrics (bias, relevance, hallucination, sentiment, toxicity). Snippets are intentionally mild and safe.

## 3. Instrumentation Modes

You can run the demo app in two modes:

### A. Zero-Code Instrumentation (Default mode)

Default mode - use OpenTelemetry auto-instrumentation launcher.

```bash
opentelemetry-instrument python main.py
```

Automatically patches supported libraries; no code changes required.

### B. Manual Instrumentation (Debug / Dev)

This mode is used for development/debugging (zero-code breaks debugger sessions in IDEs). It also may be helpful for advanced instrumentation use-cases, when zero-code instrumentation is enriched with manual instrumentation, or additional customization is needed. Use carefully, as it may break zero-code GenAI Type detection and evaluation logic.

```bash
python main.py --manual-instrumentation
```

This executes `_configure_manual_instrumentation()`, which explicitly sets SDK providers and OTLP exporters then runs `LangchainInstrumentor().instrument()`.

## 4. Sample Telemetry Trace

Below is a a demo application run trace part, where an agent invocations are being evaluated. In addition to a span, it produces evaluation result logs and metrics.

```text
Trace ID: b21415a5685ea2b2205bbb91e6f87b55
└── Span ID: 5f3240f6667ebe67 (Parent: none) - Name: gen_ai.workflow LangGraph [op:invoke_workflow] (Type: span)
    ├── Metric: gen_ai.workflow.duration (Type: metric)
    ├── Span ID: c573dabe4a0598a1 (Parent: 5f3240f6667ebe67) - Name: gen_ai.step __start__ (Type: span)
    │   └── Span ID: 1f9707fe7d183ea7 (Parent: c573dabe4a0598a1) - Name: gen_ai.step should_continue (Type: span)
    ├── Span ID: 1bb28c3a114bde71 (Parent: 5f3240f6667ebe67) - Name: gen_ai.step coordinator (Type: span)
    │   ├── Span ID: 57b954cc3a1ce66a (Parent: 1bb28c3a114bde71) - Name: invoke_agent coordinator [op:invoke_agent] (Type: span)
    │   │   ├── Log: gen_ai.client.inference.operation.details (Type: log)
    │   │   ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    │   │   ├── Metric: gen_ai.agent.duration [op:invoke_agent] (Type: metric)
    │   │   └── Span ID: d03c706160b38ebb (Parent: 57b954cc3a1ce66a) - Name: gen_ai.step model (Type: span)
    │   │       └── Span ID: 06ec07c5be271173 (Parent: d03c706160b38ebb) - Name: chat ChatOpenAI [op:chat] (Type: span)
    │   │           ├── Log: gen_ai.client.inference.operation.details [op:chat] (Type: log)
    │   │           ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    │   │           ├── Metric: gen_ai.client.operation.duration [op:chat] (Type: metric)
    │   │           ├── Metric: gen_ai.client.token.usage (input) [op:chat] (Type: metric)
    │   │           ├── Metric: gen_ai.client.token.usage (output) [op:chat] (Type: metric)
    │   │           ├── Metric: gen_ai.evaluation.bias [op:evaluation] (Type: metric)
    │   │           ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
    │   │           ├── Metric: gen_ai.evaluation.relevance [op:evaluation] (Type: metric)
    │   │           ├── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
    │   │           └── Metric: gen_ai.evaluation.toxicity [op:evaluation] (Type: metric)
    │   └── Span ID: 643e13eac446d3b2 (Parent: 1bb28c3a114bde71) - Name: gen_ai.step should_continue (Type: span)
...
```

Evaluation metrics appear on agent/model spans allowing correlation with injected poison events.

## 5. Local Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

copy and modify `.env.example` to `.env`, change there or export your `OPENAI_API_KEY`. Run the app

```bash
dotenv run -- python main.py --manual-instrumentation
```

## 6. Docker Usage

The `Dockerfile` installs dependencies and sets `CMD ["python", "main.py"]`.

Build image:

```bash
docker build -t travel-planner:latest .
```

Run manual mode:

```bash
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  travel-planner:latest python main.py --manual-instrumentation
```

Run zero-code mode (override command):

```bash
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  travel-planner:latest opentelemetry-instrument python main.py
```

## 7. Kubernetes Deployment

Given existing `k8s-cronjob.yaml`, create two CronJobs referencing the same image.

Manual instrumentation flavor:

```yaml
args: ["--manual-instrumentation"]
env:
  - name: OTEL_RESOURCE_ATTRIBUTES
    value: "deployment.environment=o11y-inframon-ai,flavor=manual"
```

Zero-code instrumentation flavor:

```yaml
command: ["opentelemetry-instrument"]
args: ["python", "main.py"]
env:
  - name: OTEL_RESOURCE_ATTRIBUTES
    value: "deployment.environment=o11y-inframon-ai,flavor=zerocode"
```

Both share the same `ConfigMap` for core settings (model, exporter flags). Distinguish telemetry in back-end via `resource.attributes.flavor`.

Run manual mode:

```bash
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
	-e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
	travel-planner:latest python main.py --manual-instrumentation
```

Run zero-code mode (override command):

```bash
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
	-e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
	travel-planner:latest opentelemetry-instrument python main.py
```

## 7. Kubernetes Deployment

Example `k8s-cronjob.yaml` runs this demo as a cron job workload

Zero-code instrumentation flavor:

```yaml
command: ["opentelemetry-instrument"]
args: ["python", "main.py"]
env:
	- name: OTEL_RESOURCE_ATTRIBUTES
	  value: "deployment.environment=o11y-inframon-ai,flavor=zerocode"
```

Manual instrumentation flavor:

```yaml
args: ["--manual-instrumentation"]
env:
	- name: OTEL_RESOURCE_ATTRIBUTES
	  value: "deployment.environment=o11y-inframon-ai,flavor=manual"
```

Both share the same `ConfigMap` for core settings (model, exporter flags). Distinguish telemetry in back-end via `resource.attributes.flavor`.

## 8. Core Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Authenticates ChatOpenAI |
| `OPENAI_MODEL` | Chooses model (default fallback in code) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint (gRPC) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Usually `grpc` |
| `OTEL_SERVICE_NAME` | Service identity |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompt & output content |
| `TRAVEL_POISON_*` | Controls prompt noise injection |

## 9. Observing Evaluation Metrics

Metrics & logs exported (names may include operation suffixes):

- `gen_ai.evaluation.hallucination`
- `gen_ai.evaluation.bias`
- `gen_ai.evaluation.relevance`
- `gen_ai.evaluation.sentiment`
- `gen_ai.evaluation.toxicity`

Higher scores often correlate with injected poison snippets.

## 10. Key Files

| File | Purpose |
|------|---------|
| `main.py` | Workflow, poisoning, optional manual instrumentation setup |
| `Dockerfile` | Container build (editable installs + example requirements) |
| `k8s-cronjob.yaml` | Example CronJob manifest (modify for two flavors) |
| `requirements.txt` | Python dependencies for the sample |

## 11. Disclaimer

Poison snippets are intentionally mild, non-harmful, and used solely to trigger evaluation telemetry for demonstration. Not production travel advice.
