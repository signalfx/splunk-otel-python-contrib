# Multi-Agent Travel Planner (LangGraph + LangChain Instrumentation)

This example demonstrates a **LangGraph-driven multi-agent workflow** for synthesizing a one-week trip itinerary using several LangChain agents. It is instrumented with the OpenTelemetry LangChain instrumentation so that spans, metrics, and logs are emitted for:

- Workflow orchestration (`gen_ai.workflow`, `gen_ai.step` spans)
- Individual agent invocations (`invoke_agent` spans & metrics)
- Underlying model calls (`chat` spans & token usage metrics)
- Automatic evaluation metrics (LLM-as-a-judge) such as hallucination, bias, relevance, sentiment, toxicity (emitted by the instrumentation layer — no evaluation code inside the app)

## Architecture Overview

Agents (implemented as lightweight ReAct-style LangChain agents):

1. `coordinator` – Interprets the traveller request and outlines plan for specialists.
2. `flight_specialist` – Selects appealing flight option (uses `mock_search_flights` tool).
3. `hotel_specialist` – Recommends boutique hotel (uses `mock_search_hotels` tool).
4. `activity_specialist` – Curates destination activities (uses `mock_search_activities` tool).
5. `plan_synthesizer` – Combines specialist outputs into final structured itinerary.

State flows through a LangGraph `StateGraph` with conditional edges driven by `should_continue`. A synthetic root HTTP span (`POST /travel/plan`) is created to represent a higher-level request context.

## Instrumentation

`LangchainInstrumentor().instrument()` wires tracing, metrics, and logs. Each agent/model call yields spans and evaluation metrics automatically (LLM-as-a-judge style). You do **not** need to add any evaluation code — the instrumentation inspects messages and model responses.

## Prompt Poisoning (Quality Degradation) Mechanism

To showcase evaluation metrics being triggered, the application can *randomly inject quality-degrading snippets* into agent prompts. This simulates issues like hallucination, mild bias, irrelevance, negative sentiment, or mild toxicity. The injection happens inside helpers:

- `_poison_config()` reads environment-driven configuration.
- `maybe_add_quality_noise()` conditionally appends one or more poison snippets to the prompt for the current agent.
- Injected events are recorded on state as `poison_events` and exported on the root span attribute `travel.plan.poison_events` for correlation.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAVEL_POISON_PROB` | `0.35` | Probability (0–1) that a given agent step is poisoned. |
| `TRAVEL_POISON_TYPES` | `hallucination,bias,irrelevance,negative_sentiment,toxicity` | Comma-separated subset of supported poison kinds. |
| `TRAVEL_POISON_MAX` | `2` | Maximum number of snippets injected per poisoned step. |
| `TRAVEL_POISON_SEED` | *unset* | Optional deterministic seed for reproducibility. |

Each poison kind maps to a short, non-harmful snippet that is *likely* to influence evaluator dimensions:

- `hallucination` – Introduces impossible infrastructure (e.g., underwater hyperloop).
- `bias` – Pushes preference for luxury despite sustainability concerns.
- `irrelevance` – Requests unrelated technical aside (quantum acronyms).
- `negative_sentiment` – Adds discouraging framing about traveller expectations.
- `toxicity` – Mildly caustic remark about coordination process.

The snippets are intentionally mild and safe while still altering qualitative characteristics.

## Running the Example

Ensure you have the required environment variables for an OpenAI-compatible model (e.g., `OPENAI_API_KEY`) set, plus any OTLP exporter configuration if you want to collect telemetry.

```bash
# (optional) Reproducible run with deterministic poisoning
export TRAVEL_POISON_SEED=42
export TRAVEL_POISON_PROB=0.75
export TRAVEL_POISON_TYPES=hallucination,bias,irrelevance
python main.py
```

After execution, spans and metrics (including evaluation metrics) are sent to the configured OTLP endpoint. The process sleeps briefly at the end to allow asynchronous exporters to flush.

## Observing Evaluation Metrics

Look for metrics/logs named:

- `gen_ai.evaluation.hallucination`
- `gen_ai.evaluation.bias`
- `gen_ai.evaluation.relevance`
- `gen_ai.evaluation.sentiment`
- `gen_ai.evaluation.toxicity`

Poisoning increases the likelihood these metrics surface anomalies or elevated scores.

## Extending the Scenario (Ideas)

You can further enrich realism without adding evaluation code:

1. Dynamic tool agent: Introduce a new `budget_checker` or `sustainability_auditor` agent whose prompt is frequently poisoned to simulate inconsistent tool use.
2. Structured output stress: Force the `plan_synthesizer` to produce strict JSON, then occasionally poison the prompt causing structural drift (triggering relevance or hallucination). *Leverage LangChain structured output APIs.*
3. Memory deviations: Integrate short-term memory (see LangChain memory docs) and occasionally inject contradictory memory entries.
4. Streaming partial hallucinations: Switch to streaming responses and randomly append a poison snippet mid-stream.
5. Middleware insertion: Use LangChain middleware to mutate messages post-agent reasoning for a separate layer of noise.

## Key Files

- `main.py` – Application entry point and workflow definition.
- This `README.md` – Documentation of architecture and poisoning mechanism.

## Telemetry Root Span Attributes

- `travel.plan.preview` – Truncated preview of final itinerary.
- `travel.plan.poison_events` – CSV list of injected poison events (`agent:kind`).
- `travel.session_id` – Synthetic session correlation ID.
- `travel.agents_used` – Count of specialist agents that produced content.

## Disclaimer

The poisoning snippets are intentionally constrained to avoid harmful, hateful, or sensitive content. They exist solely to exercise evaluation dimensions and should not be considered realistic travel advice.

---
Happy tracing! Observe how even mild prompt perturbations surface in evaluation telemetry.
