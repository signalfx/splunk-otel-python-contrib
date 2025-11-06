"""Weather Agent Demo v1 (Real OpenAI + Open-Meteo)

Features:
    * Real OpenAI chat model usage (if OPENAI_API_KEY set; otherwise echo fallback)
    * Structured output parsing via LangChain provider structured output (WeatherQuery)
    * Multi-agent style supervisor (subagent wrappers as tools)
    * LangGraph workflow alternative with parallel fetch workers
    * Real external data calls (geocoding + forecast + air quality) via Open-Meteo APIs (no API key required)
    * Activity advisory computation logic

Run:
    python weather-agent-demo-v1.py "Weather for Paris tomorrow and is it good for a morning run? Also compare with Berlin"
"""
# pyright: ignore
# mypy: ignore-errors
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict, Tuple, Annotated

import requests
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
)

provider = TracerProvider()
processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Set environment variables for LangChain
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "<export-yours>" 
# os.environ["LANGSMITH_OTEL_EXPORTER"] = "otlp"
# os.environ["LANGSMITH_OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318/v1/traces"
# os.environ["LANGSMITH_API_KEY"] = ""

# ---------------------------------------------------------------------------
# GenAI Utils
# ---------------------------------------------------------------------------
# from opentelemetry import _events, _logs, metrics, trace
# from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
# from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
#     OTLPMetricExporter,
# )
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
#     OTLPSpanExporter,
# )
# from opentelemetry.instrumentation.langchain import LangchainInstrumentor
# from opentelemetry.sdk._events import EventLoggerProvider
# from opentelemetry.sdk._logs import LoggerProvider
# from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
# from opentelemetry.sdk.metrics import MeterProvider
# from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor

# # Configure tracing/metrics/logging once per process so exported data goes to OTLP.
# trace.set_tracer_provider(TracerProvider())
# trace.get_tracer_provider().add_span_processor(
#     BatchSpanProcessor(OTLPSpanExporter())
# )

# demo_tracer = trace.get_tracer("instrumentation.langchain.demo")

# metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
# metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# _logs.set_logger_provider(LoggerProvider())
# _logs.get_logger_provider().add_log_record_processor(
#     BatchLogRecordProcessor(OTLPLogExporter())
# )
# _events.set_event_logger_provider(EventLoggerProvider())

# from opentelemetry.instrumentation.langchain import LangchainInstrumentor
# instrumentor = LangchainInstrumentor()
# instrumentor.instrument()


# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------
class WeatherQuery(BaseModel):
    locations: List[str] = Field(..., description="List of locations requested")
    units: Literal["metric", "imperial"] = Field("metric")
    days: int = Field(1, ge=1, le=5, description="Number of forecast days")
    include_air_quality: bool = Field(True)
    include_sun: bool = Field(True)
    activity: Optional[str] = Field(None, description="Activity of interest (e.g. run)")

class RawWeatherData(BaseModel):
    location: str
    current_temp: float
    forecast: List[Dict[str, Any]]
    air_quality_index: Optional[int] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None
    source: str = "open-meteo"

class ActivityAdvisory(BaseModel):
    location: str
    run_ok: bool
    note: str

class ComparativeSummary(BaseModel):
    primary: str
    diff: List[str]
    highlights: List[str]

class WeatherReport(BaseModel):
    query: WeatherQuery
    data: List[RawWeatherData]
    advisories: List[ActivityAdvisory]
    comparison: Optional[ComparativeSummary]
    generated_at: datetime

def get_model() -> BaseChatModel:
    """Initialize chat model. Requires OPENAI_API_KEY set. Raises if unavailable."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; please export it to run this demo with a real model.")
    return init_chat_model("openai:gpt-4o-mini")

MODEL = get_model()

def parse_weather_query(user_text: str, run_config: Optional[Dict[str, Any]] = None) -> WeatherQuery:
    """Parse user query into WeatherQuery using structured output if available; else heuristic fallback.
    Pass run_config to propagate unified trace metadata.
    """
    try:
        if hasattr(MODEL, "with_structured_output"):
            pm = MODEL.with_structured_output(WeatherQuery)  # type: ignore[attr-defined]
            # LangChain model.invoke signature supports config param for tracing context.
            msg = pm.invoke([
                SystemMessage("Extract weather query parameters into WeatherQuery schema."),
                HumanMessage(user_text),
            ], config=run_config)  # type: ignore[arg-type]
            if isinstance(msg, WeatherQuery):
                return msg
            if isinstance(msg, AIMessage):
                try:
                    if isinstance(msg.content, str):
                        return WeatherQuery(**json.loads(msg.content))
                except Exception:
                    pass
    except Exception:
        pass
    tokens = [t.strip("?,.") for t in user_text.split()]
    locs: List[str] = []
    composite: List[str] = []
    for i, t in enumerate(tokens):
        title = t.title()
        if title in ["Paris", "Berlin", "London", "Tokyo", "Rome", "Madrid", "New", "York"]:
            if title == "New" and i + 1 < len(tokens) and tokens[i + 1].title() == "York":
                composite.append("New York")
            elif title != "New":
                locs.append(title)
    if composite:
        locs.extend(composite)
    if not locs:
        locs = ["Paris"]
    return WeatherQuery(locations=list(dict.fromkeys(locs)), units="metric", days=1,
                        include_air_quality=True, include_sun=True, activity="run")

CACHE: Dict[str, RawWeatherData] = {}
COORD_CACHE: Dict[str, Tuple[float, float]] = {}

def _safe_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} for {url} params={params}")
    return resp.json()

# ---------------------------------------------------------------------------
# Tools (decorated)
# ---------------------------------------------------------------------------
def _call(tool_obj: Any, *args: Any, run_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    """Invoke a StructuredTool or plain function uniformly.
    Robust argument mapping for StructuredTool:
      1. If the tool exposes .args_schema (pydantic model), map positional args to field order.
      2. Merge kwargs over positional mapping (kwargs win).
      3. If only one positional arg and only one parameter, pass the raw value.
      4. Provide config when run_config supplied.
    Falls back to underlying .func or direct call if not a StructuredTool.
    """
    is_structured = hasattr(tool_obj, "invoke") and callable(getattr(tool_obj, "invoke"))
    if is_structured:
        # Build input payload
        payload: Any
        arg_model = getattr(tool_obj, "args_schema", None)
        if arg_model and hasattr(arg_model, "model_fields"):
            field_names = list(arg_model.model_fields.keys())  # type: ignore[attr-defined]
            if len(field_names) == 1 and len(args) == 1 and not kwargs:
                # Single parameter tool; allow raw value
                payload = args[0]
            else:
                mapped: Dict[str, Any] = {}
                for idx, val in enumerate(args):
                    if idx < len(field_names):
                        fname = field_names[idx]
                        # Avoid overwriting explicit kwarg
                        if fname not in kwargs:
                            mapped[fname] = val
                # Special-case geocode if 'query' not set and first arg exists
                if 'geocode' in str(getattr(tool_obj, 'name', '')) and 'query' not in kwargs and 'query' not in mapped and args:
                    mapped['query'] = args[0]
                # Merge kwargs last
                mapped.update(kwargs)
                payload = mapped
        else:
            # No args schema; best-effort: if kwargs present and args present map first arg to common names
            if args and kwargs:
                merged = dict(kwargs)
                common_keys = ["location", "query", "input"]
                for ck in common_keys:
                    if ck not in merged and args:
                        merged[ck] = args[0]
                        break
                payload = merged
            elif kwargs:
                payload = kwargs
            elif len(args) == 1:
                payload = args[0]
            else:
                payload = list(args)
        if run_config is not None:
            return tool_obj.invoke(payload, config=run_config)  # type: ignore[attr-defined]
        return tool_obj.invoke(payload)  # type: ignore[attr-defined]
    # Non-structured fallback
    if hasattr(tool_obj, "func") and callable(getattr(tool_obj, "func")):
        return tool_obj.func(*args, **kwargs)  # type: ignore[attr-defined]
    return tool_obj(*args, **kwargs)
@tool
def geocode_location(query: str) -> str:
    """Geocode a location using Open-Meteo geocoding API; cache coordinates and return canonical name."""
    q = query.strip()
    try:
        data = _safe_request("https://geocoding-api.open-meteo.com/v1/search", {"name": q, "count": 1})
        results = data.get("results") or []
        if results:
            r0 = results[0]
            name = r0.get("name")
            lat = r0.get("latitude")
            lon = r0.get("longitude")
            if isinstance(name, str) and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                COORD_CACHE[name] = (float(lat), float(lon))
                return name
    except Exception:
        pass
    fallback = q.title().split(",")[0]
    COORD_CACHE.setdefault(fallback, (48.8566, 2.3522))  # Paris default
    return fallback

@tool
def fetch_weather(location: str, units: str = "metric", days: int = 1) -> RawWeatherData:
    """Fetch weather forecast (current + daily sunrise/sunset) via Open-Meteo API."""
    if location in CACHE:
        return CACHE[location]
    if location not in COORD_CACHE:
        _call(geocode_location, location)
    lat, lon = COORD_CACHE[location]
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "daily": "sunrise,sunset",
        "forecast_days": days,
        "timezone": "UTC",
    }
    data = _safe_request("https://api.open-meteo.com/v1/forecast", params)
    current = data.get("current_weather", {}).get("temperature")
    sunrise_list = data.get("daily", {}).get("sunrise", [])
    sunset_list = data.get("daily", {}).get("sunset", [])
    forecast: List[Dict[str, Any]] = []
    for i in range(days):
        forecast.append({
            "day": i + 1,
            "temp": current,
            "sunrise": sunrise_list[i] if i < len(sunrise_list) else None,
            "sunset": sunset_list[i] if i < len(sunset_list) else None,
        })
    w = RawWeatherData(
        location=location,
        current_temp=float(current) if current is not None else 0.0,
        forecast=forecast,
        sunrise=sunrise_list[0] if sunrise_list else None,
        sunset=sunset_list[0] if sunset_list else None,
    )
    CACHE[location] = w
    return w

@tool
def fetch_air_quality(location: str) -> int:
    """Fetch air quality (US AQI) from Open-Meteo air-quality API."""
    if location not in COORD_CACHE:
        _call(geocode_location, location)
    lat, lon = COORD_CACHE[location]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi",
        "timezone": "UTC",
    }
    try:
        data = _safe_request("https://air-quality-api.open-meteo.com/v1/air-quality", params)
        aqi_series = data.get("hourly", {}).get("us_aqi", [])
        return int(aqi_series[0]) if aqi_series else 0
    except Exception:
        return 0

@tool
def fetch_sun_times(location: str) -> Dict[str, str]:
    """Return sunrise/sunset from cached forecast (avoids extra API calls)."""
    if location not in CACHE:
        _call(fetch_weather, location)
    d = CACHE[location]
    return {"sunrise": d.sunrise or "", "sunset": d.sunset or ""}

@tool
def generate_activity_advice(location: str, temp: float, aqi: Optional[int] = None) -> ActivityAdvisory:
    """Generate running suitability advisory."""
    run_ok = (temp >= 5 and temp <= 30) and (aqi is None or aqi < 100)
    note = f"Run OK: {run_ok}. Temp={temp}C AQI={aqi}" if aqi is not None else f"Run OK: {run_ok}. Temp={temp}C"
    return ActivityAdvisory(location=location, run_ok=run_ok, note=note)

@tool
def compare_locations(locations: List[str], data: Dict[str, Any]) -> ComparativeSummary:
    """Compare first location with others."""
    if not locations:
        raise ValueError("No locations provided")
    primary = locations[0]
    diff = locations[1:]
    highlights = [f"{loc}: temp={data[loc].current_temp}" for loc in locations if loc in data]
    return ComparativeSummary(primary=primary, diff=diff, highlights=highlights)

@tool
def summarize_conversation(runtime: Any) -> str:  # type: ignore
    """Summarize message counts from runtime state."""
    state_msgs = runtime.state["messages"] if "messages" in runtime.state else []  # type: ignore
    human = sum(1 for m in state_msgs if getattr(m, "type", "") == "human")
    ai = sum(1 for m in state_msgs if getattr(m, "type", "") == "ai")
    tool_msgs = sum(1 for m in state_msgs if getattr(m, "type", "") == "tool")
    return f"Conversation counts human={human} ai={ai} tool={tool_msgs}"

# ---------------------------------------------------------------------------
# Multi-agent supervisor setup (subagents as tools)
# ---------------------------------------------------------------------------
# Each subagent will simply call underlying tools directly; for demo we simulate via wrappers.

# For brevity we treat subagents as simple passthrough callables decorated with @tool
@tool("call_forecast")
def call_forecast(location: str, units: str = "metric", days: int = 1) -> str:
    """Subagent: get weather + sun times for a location"""
    w = _call(fetch_weather, location, units=units, days=days)
    sun = _call(fetch_sun_times, location)
    return json.dumps({"weather": w.dict(), "sun": sun})

@tool("call_air_quality")
def call_air_quality(location: str) -> str:
    """Subagent: get air quality for a location"""
    aqi = _call(fetch_air_quality, location)
    return json.dumps({"aqi": aqi})

@tool("call_recommendation")
def call_recommendation(location: str, temp: float, aqi: Optional[int] = None) -> str:
    """Subagent: running recommendation for location"""
    adv = _call(generate_activity_advice, location, temp=temp, aqi=aqi)
    return adv.json()

SUPERVISOR_TOOLS = [
    geocode_location,
    call_forecast,
    call_air_quality,
    call_recommendation,
    compare_locations,
    summarize_conversation,
]

SUPERVISOR_AGENT = create_agent(  # type: ignore
    name="weather_supervisor_agent",
    model=MODEL,
    tools=SUPERVISOR_TOOLS,
    system_prompt="You are a weather supervisor agent. Use tools to gather data and produce concise results.",
).with_config({
    "run_name": "weather-agent",
    "tags": ["agent:supervisor", "agent"],
    "metadata": {"agent_name": "supervisor", "agent_role": "orchestrator"}
})

# ---------------------------------------------------------------------------
# LangGraph workflow implementation
# ---------------------------------------------------------------------------
def _merge_data_maps(existing: Dict[str, RawWeatherData], new: Dict[str, RawWeatherData]) -> Dict[str, RawWeatherData]:
    """Aggregator for parallel worker_fetch node outputs.
    LangGraph calls this when multiple updates target the same key in the same step.
    """
    merged = dict(existing)
    merged.update(new)
    return merged

class GraphState(TypedDict):
    input: str
    query: Optional[WeatherQuery]
    locations: List[str]
    # Annotated with merge function so parallel updates accumulate rather than conflict
    data_map: Annotated[Dict[str, RawWeatherData], _merge_data_maps]
    advisories: List[ActivityAdvisory]
    comparison: Optional[ComparativeSummary]
    report: Optional[WeatherReport]

def node_parse(state: GraphState) -> Dict[str, Any]:
    q = parse_weather_query(state["input"])
    return {"query": q, "locations": q.locations}

def node_parallel_fetch(state: GraphState):
    return [Send("worker_fetch", {"location": loc}) for loc in state["locations"]]

def worker_fetch(state: Dict[str, Any]):
    loc = state["location"]
    w = _call(fetch_weather, loc, units="metric", days=1)
    aqi = _call(fetch_air_quality, loc)
    sun = _call(fetch_sun_times, loc)
    w.air_quality_index = aqi
    w.sunrise = sun["sunrise"]
    w.sunset = sun["sunset"]
    return {"data_map": {loc: w}}

def node_advisory(state: GraphState):
    advisories = [
        _call(generate_activity_advice, loc, w.current_temp, w.air_quality_index)
        for loc, w in state["data_map"].items()
    ]
    return {"advisories": advisories}

def node_compare(state: GraphState):
    if len(state["data_map"]) > 1:
        cmp_summary = _call(compare_locations, list(state["data_map"].keys()), state["data_map"])
        return {"comparison": cmp_summary}
    return {}

def node_finalize(state: GraphState):
    report = WeatherReport(
        query=state["query"],
        data=list(state["data_map"].values()),
        advisories=state.get("advisories", []),
        comparison=state.get("comparison"),
        generated_at=datetime.now(timezone.utc),
    )
    return {"report": report}

workflow = StateGraph(GraphState)
workflow.add_node("parse", node_parse)
workflow.add_node("worker_fetch", worker_fetch)
workflow.add_node("advisory", node_advisory)
workflow.add_node("compare", node_compare)
workflow.add_node("finalize", node_finalize)
workflow.add_edge(START, "parse")
workflow.add_conditional_edges("parse", node_parallel_fetch, ["worker_fetch"])
workflow.add_edge("worker_fetch", "advisory")
workflow.add_edge("advisory", "compare")
workflow.add_edge("compare", "finalize")
workflow.add_edge("finalize", END)
GRAPH_APP = workflow.compile()

# ---------------------------------------------------------------------------
# Helper to run supervisor agent (single turn for demo)
# ---------------------------------------------------------------------------

def run_supervisor(user_query: str, run_config: Dict[str, Any]) -> WeatherReport:
    """Run supervisor path using SUPERVISOR_AGENT for orchestration and instrument tool calls under same run_id."""
    # Agent invocation (produces top-level run in trace)
    SUPERVISOR_AGENT.invoke({"messages": [HumanMessage(user_query)]}, config=run_config)  # type: ignore
    parsed = parse_weather_query(user_query, run_config=run_config)
    data_items: List[RawWeatherData] = []
    for loc in parsed.locations:
        canonical = _call(geocode_location, loc, run_config=run_config)
        w = _call(fetch_weather, canonical, units=parsed.units, days=parsed.days, run_config=run_config)
        aqi = _call(fetch_air_quality, canonical, run_config=run_config) if parsed.include_air_quality else None
        if aqi is not None:
            w.air_quality_index = aqi
        # Sun times captured inside fetch_weather forecast; augment if needed
        sun = _call(fetch_sun_times, canonical, run_config=run_config)
        w.sunrise = sun.get("sunrise") or w.sunrise
        w.sunset = sun.get("sunset") or w.sunset
        data_items.append(w)
    advisories = [
        _call(
            generate_activity_advice,
            d.location,
            d.current_temp,
            d.air_quality_index if parsed.include_air_quality else None,
            run_config=run_config,
        )
        for d in data_items
    ]
    comparison = _call(
        compare_locations,
        parsed.locations,
        {d.location: d for d in data_items},
        run_config=run_config,
    ) if len(parsed.locations) > 1 else None
    return WeatherReport(
        query=parsed,
        data=data_items,
        advisories=advisories,
        comparison=comparison,
        generated_at=datetime.now(timezone.utc),
    )

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    graph_only = False
    if "--graph-only" in args:
        graph_only = True
        args = [a for a in args if a != "--graph-only"]
    query = args[0] if args else "Weather for Paris tomorrow and is it good for a morning run? Also compare with Berlin"

    # Unified run_id for LangChain/LangSmith trace across supervisor + graph
    run_id = str(uuid.uuid4())
    unified_config = {"run_id": run_id, "tags": ["weather-workflow", "supervisor"]}

    if not graph_only:
        print("=== Supervisor Deterministic Path (Unified Trace) ===")
        sup_report = run_supervisor(query, run_config=unified_config)
        print(json.dumps(sup_report.model_dump(), indent=2, default=str))

    print("\n=== LangGraph Workflow Path (stream updates) ===")
    init_state: GraphState = {
        "input": query,
        "query": None,
        "locations": [],
        "data_map": {},
        "advisories": [],
        "comparison": None,
        "report": None,
    }
    graph_config = {"run_id": run_id, "tags": ["weather-workflow", "graph"]}
    for chunk in GRAPH_APP.stream(init_state, stream_mode="updates", config=graph_config):  # type: ignore[arg-type]
        for node_name, node_state in chunk.items():
            print(f"[update] node={node_name} keys={list(node_state.keys())}")
    final = GRAPH_APP.invoke(init_state, config=graph_config)
    report = final.get("report")
    if report:
        print("\nFinal Graph Report:")
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print("Graph failed to produce report.")

    print("\nDone.")


if __name__ == "__main__":
    main()
