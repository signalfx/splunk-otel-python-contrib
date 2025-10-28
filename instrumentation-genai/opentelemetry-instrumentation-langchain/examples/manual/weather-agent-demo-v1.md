# Weather Agent Demo (v1) – Implementation Overview

This document describes the implementation in `weather-agent-demo-v1.py` for a LangChain + LangGraph weather demo instrumented with OpenTelemetry.

## Demo App scope

* Deterministic supervisor orchestration (no autonomous agent loop yet).
* Pydantic models: `WeatherQuery`, `RawWeatherData`, `ActivityAdvisory`, `ComparativeSummary`, `WeatherReport`.
* Structured query parsing with provider structured output attempt + heuristic fallback.
* Tool definitions using `@tool` decorator, invoked safely via `_call` wrapper for `StructuredTool`.
* LangGraph workflow (conditional execution + parallel fetch) guarded by import availability.
* Concurrency for parallel node updates using `Annotated` merge aggregator on `data_map`.
* OpenTelemetry spans around parsing, tools, supervisor run, and graph nodes.
* External API calls (Open-Meteo geocoding, forecast, air quality).
* CLI flag `--graph-only` to exercise graph path without supervisor.

### Out of scope

short-term memory; streaming token output; middleware examples; pytest suite; structured output retry layer.

## Official Documentation References

### LangChain v1

* [Overview](https://docs.langchain.com/oss/python/langchain/overview)
* [Models](https://docs.langchain.com/oss/python/langchain/models)
* [Messages](https://docs.langchain.com/oss/python/langchain/messages)
* [Tools](https://docs.langchain.com/oss/python/langchain/tools)
* [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
* [Multi-Agent](https://docs.langchain.com/oss/python/langchain/multi-agent)
* [Short-Term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
* [Streaming](https://docs.langchain.com/oss/python/langchain/streaming)
* [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)

### LangGraph v1

* [Overview](https://docs.langchain.com/oss/python/langgraph/overview)
* [Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
* [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)

## File Overview

* Imports + graceful fallbacks for LangChain/LangGraph.
* Pydantic data models.
* Model selection (`_get_model`) chooses OpenAI GPT-4o-mini if `OPENAI_API_KEY` else echo stub.
* Query parsing (`parse_weather_query`).
* HTTP helpers and caches (`_safe_get`, `COORD_CACHE`, `WEATHER_CACHE`).
* Plain implementations and `@tool` wrappers.
* Supervisor orchestration (`run_supervisor`) using `_call`.
* LangGraph workflow definition (`GraphState`, nodes, parallel `Send`).
* CLI entry (`main`) with `--graph-only` flag.

## Data Models

* `WeatherQuery` – user intent (locations, units, days, flags, activity).
* `RawWeatherData` – temperature, forecast slice, sunrise/sunset, optional AQI.
* `ActivityAdvisory` – running suitability (boolean + note).
* `ComparativeSummary` – differences & highlights across locations.
* `WeatherReport` – aggregate result + UTC timestamp.

## Query Parsing Strategy

Attempts `model.with_structured_output(WeatherQuery)`; if provider or parsing fails, falls back to heuristic token scan (supports multi-word “New York”). Defaults to Paris when no known location found; activity defaults to `run`.

## Tool Invocation Abstraction

`_call(tool_obj, *args, **kwargs)` prevents `TypeError: 'StructuredTool' object is not callable`:

* Use underlying `.func` when present.
* Fallback to `.invoke()` with kwarg adaptation.
* Else call directly.


## Implemented Tools

* `geocode_location(query)` – canonical name + cache coordinates.
* `fetch_weather(location, units, days)` – forecast + sunrise/sunset.
* `fetch_air_quality(location)` – first hourly US AQI value.
* `fetch_sun_times(location)` – derive from cached forecast.
* `generate_activity_advice(location, temp, aqi?)` – threshold logic.
* `compare_locations(locations, data)` – build `ComparativeSummary`.
* Subagent-style wrappers: `call_forecast`, `call_air_quality`, `call_recommendation`.

## Supervisor Execution Flow

1. Parse query → `WeatherQuery`.
2. For each location: geocode, weather, optional AQI.
3. Build advisories.
4. Optional comparison.
5. Return `WeatherReport`.

Span: `run.supervisor` + per-tool spans (`tool.geocode`, etc.).

## LangGraph Workflow

Nodes (compiled only if LangGraph available):

* `parse` – extract `query`, `locations`.
* Parallel `worker_fetch` – each emits partial `data_map`.
* `advisory` – create `ActivityAdvisory` list.
* `compare` – conditional summary if >1 location.
* `finalize` – construct `WeatherReport`.


### Concurrency Aggregation Fix

Parallel `worker_fetch` writes caused `InvalidUpdateError`. Resolved with annotated merge aggregator:
 
```python
def _merge_data_maps(existing: Dict[str, RawWeatherData], new: Dict[str, RawWeatherData]) -> Dict[str, RawWeatherData]:
    if existing is None:
        return dict(new)
    merged = dict(existing)
    merged.update(new)
    return merged
```

Applied via `Annotated[Dict[str, RawWeatherData], _merge_data_maps]` in `GraphState`.

## Instrumentation (OpenTelemetry)

Minimal spans:

* `model.parse_query`
* `tool.*` for each tool call
* `run.supervisor`
* Graph node spans (e.g., `graph.node.parse` or inline)



Attributes: input length, location, cache hit, AQI. Future: add durations, retry counts, error codes.

## Running the Demo

Supervisor + graph path:

```bash
python weather-agent-demo-v1.py "Weather for Paris tomorrow and is it good for a morning run? Also compare with Berlin"
```

Graph only:

```bash
python weather-agent-demo-v1.py --graph-only "Weather for Paris compare with Berlin"
```

Without `OPENAI_API_KEY`, an echo model fallback is used.

## Structured Output Behavior

Returns a Pydantic instance directly when provider structured output succeeds; otherwise consumes JSON from `AIMessage` or falls back to heuristic parsing.

## Error Handling & Resilience

* API failures tagged (`fetch.error`, `aqi.error`) with fallback values.
* Geocode fallback uses title case + Paris coordinate default if unknown.
* Missing AQI → advisory still computed.
* Structured output parsing failures gracefully degrade.

## Pending / Future Enhancements

* Short-term memory & preference injection.
* Streaming tokens for advisory reasoning.
* Middleware (logging, retry/backoff, dynamic prompt enrichment).
* Pytest suite (parsing, tools, concurrency aggregator, advisories extremes).
* Structured output validation & retry loop.
* Conditional AQI fetch branch in graph.
* Extended activity recommendations (cycling, hiking, indoor fallback).
* OTLP exporter / rich console output for spans & metrics.

## Design Trade-offs

* Deterministic supervisor simplifies instrumentation vs. autonomous agent loops.
* Minimal graph avoids premature complexity before memory/middleware integration.
* Streaming deferred until core stability (invocation + concurrency) established.

## Quick Reference

* Models: `WeatherQuery`, `WeatherReport`
* Helpers: `_call`, `parse_weather_query`, `run_supervisor`
* Tools: `geocode_location`, `fetch_weather`, `fetch_air_quality`, `fetch_sun_times`, `generate_activity_advice`, `compare_locations`
* Graph nodes: `node_parse`, `worker_fetch`, `node_advisory`, `node_compare`, `node_finalize`
* Aggregator: `_merge_data_maps`

## Next Steps

1. Implement memory layer.
2. Add streaming output path.
3. Integrate middleware examples.
4. Add tests & CI.
5. Expand graph with distinct AQI/sun worker nodes.
6. Add structured output retry strategy.

## License / Attribution

Uses Open-Meteo public APIs ([open-meteo.com](https://open-meteo.com/)) and integrates LangChain / LangGraph per referenced documentation. OpenTelemetry instrumentation follows general tracing best practices.

---
This document reflects the current runnable implementation and replaces the earlier planning draft.
