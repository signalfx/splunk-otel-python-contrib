# Multi-Agent Travel Planner (Traceloop translator + OpenTelemetry)

This example shows a small team of LangChain agents connected by a LangGraph state machine to produce a week-long city break itinerary (flights, hotel, activities, synthesis). It demonstrates GenAI observability (traces, metrics, logs, evaluation metrics) using the Traceloop SDK. 

Also, for [more info](https://help.splunk.com/en/splunk-observability-cloud/observability-for-ai/instrument-an-ai-application/collect-data-from-traceloop-instrumented-ai-applications)

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

## 3. Local Setup

```bash
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/traceloop
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

modify `.env`, change there or export your `OPENAI_API_KEY`. Run the app

Evaluation metrics appear on `model/LLM` spans allowing correlation with injected poison events. These synthetic spans are added in `on_end()` lifecycle of the span using the `TelemetryHandelr` in `splunk-otel-util-genai-translator-traceloop`

## 4. Docker Usage

The `Dockerfile` installs dependencies and sets `CMD ["python3", "main.py"]`.

Build image:

```bash
docker build -t travel-planner-tl:latest .
```

Run manual mode:

```bash
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  travel-planner-tl:latest python main.py
```

## 7. Kubernetes Deployment

Example `cronjob.yaml` runs this demo as a cron job workload

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
| `cronjob.yaml` | Example CronJob manifest (modify for two flavors) |
| `requirements.txt` | Python dependencies for the sample |

## 11. Disclaimer

Poison snippets are intentionally mild, non-harmful, and used solely to trigger evaluation telemetry for demonstration. Not production travel advice.
