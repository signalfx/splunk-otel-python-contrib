"""Multi-Agent Travel Planner — FastAPI + LangGraph served by Gunicorn + Uvicorn workers.

Overview
────────
A five-agent LangGraph pipeline that produces a complete travel itinerary:

  coordinator → flight_specialist → hotel_specialist
              → activity_specialist → plan_synthesizer

Each agent is an independent LLM call (Azure OpenAI or standard OpenAI) tagged
with ``gen_ai.agent.name`` metadata so that per-agent telemetry is visible in
Splunk Observability Cloud's APM Agent view.

Telemetry emitted (via splunk-otel-instrumentation-langchain)
──────────────────────────────────────────────────────────────
Spans:
  - One ``gen_ai.agent.invoke`` span per LangGraph node (coordinator,
    flight_specialist, hotel_specialist, activity_specialist, plan_synthesizer)
  - One ``gen_ai.client.chat`` span per LLM call inside each node

Metrics (delta temporality):
  - ``gen_ai.client.token.usage``   — prompt / completion tokens per agent
  - ``gen_ai.client.operation.duration`` — latency histogram per operation

All signals carry ``deployment.environment``, ``service.name``,
``gen_ai.agent.name``, and ``gen_ai.request.model`` attributes.

OTel initialisation strategy
─────────────────────────────
Two approaches are supported and can be selected at startup time:

  A) CLI auto-instrumentation (zero-code, recommended for Gunicorn + UvicornWorker):

       opentelemetry-instrument gunicorn -w N -k uvicorn.workers.UvicornWorker app:app

     UvicornWorker is fork-aware and preserves background threads, so all three
     signals (traces, metrics, logs) flow correctly without extra configuration.
     Reference: https://opentelemetry.io/docs/zero-code/python/troubleshooting/#pre-fork-server-issues

  B) Programmatic auto-instrumentation (used in this file / Azure App Service):

       gunicorn -w N -k uvicorn.workers.UvicornWorker app:app  # startup.sh

     ``initialize()`` is called at module import time, inside each worker process
     after the Gunicorn ``fork()``.  This ensures the ``PeriodicExportingMetricReader``
     background thread starts in the right process, preventing the silent metric
     drop that occurs when the SDK is initialised in the master process and then
     inherited across a fork (Linux).
     Reference: https://opentelemetry.io/docs/zero-code/python/troubleshooting/#use-programmatic-auto-instrumentation

Guard logic (lines below the docstring):
  1. ``sys.modules`` sentinel — prevents re-running ``initialize()`` if app.py is
     reimported within the same process (hot-reload, test collection).
  2. ``TracerProvider`` type check — skips ``initialize()`` when the CLI wrapper
     (approach A) has already set up a real SDK provider in this process, so both
     approaches coexist cleanly without double-initialisation.

Support matrix (Gunicorn + UvicornWorker, verified on Azure App Service):
  Approach        Workers  Traces  Metrics  Logs  Notes
  CLI wrapper       1        ✓       ✓        ✓
  CLI wrapper       N        ✓       ✓        ✓   UvicornWorker handles fork safety
  Programmatic      1        ✓       ✓        ✓
  Programmatic      N        ✓       ✓        ✓   initialize() runs post-fork per worker

Verified deployments
─────────────────────
• Local:         uvicorn app:app --reload  (development)
• Local:         gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app
• Azure App Svc: Python 3.12 runtime, startup.sh, Splunk OTel Collector 0.123.0 in ACI
  - Traces visible in Splunk APM Trace Analyzer
  - Metrics (token usage, operation duration) visible in Splunk APM Agent view

Required environment variables
────────────────────────────────
  AZURE_OPENAI_ENDPOINT            Azure OpenAI resource endpoint
  AZURE_OPENAI_API_KEY             Azure OpenAI API key
  AZURE_OPENAI_DEPLOYMENT          Model deployment name (e.g. gpt-4o-mini)
  AZURE_OPENAI_API_VERSION         API version (e.g. 2024-02-01)

  OTEL_SERVICE_NAME                Service name shown in Splunk
  OTEL_RESOURCE_ATTRIBUTES         e.g. deployment.environment=my-env
  OTEL_EXPORTER_OTLP_ENDPOINT      OTel Collector endpoint (e.g. http://host:4317)
  OTEL_EXPORTER_OTLP_PROTOCOL      grpc  (recommended)
  OTEL_METRICS_EXPORTER            otlp
  OTEL_TRACES_EXPORTER             otlp
  OTEL_LOGS_EXPORTER               otlp
  OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE  delta
  OTEL_INSTRUMENTATION_GENAI_EMITTERS                span_metric
  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT SPAN_ONLY

Test:
    curl -X POST http://localhost:8000/plan \\
        -H "Content-Type: application/json" \\
        -d '{"origin":"Seattle","destination":"Tokyo","travellers":2}'
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Programmatic OTel initialisation — approach B (see docstring above).
#
# Guard logic:
#   1. sys.modules sentinel: prevents re-running if app.py is re-imported
#      within the same process (e.g. hot-reload, test collection).
#   2. SDK provider check: skips initialize() when the CLI wrapper
#      (opentelemetry-instrument) has already set up a real TracerProvider
#      in this process.  This lets both approach A and B coexist cleanly:
#      - approach A: CLI inits first → SDK provider present → skip here
#      - approach B: no CLI → proxy provider present → initialize() runs here
# ---------------------------------------------------------------------------
_OTEL_INIT_KEY = "__travel_planner_otel_initialized__"
if _OTEL_INIT_KEY not in sys.modules:
    sys.modules[_OTEL_INIT_KEY] = True  # type: ignore[assignment]
    try:
        from opentelemetry import trace as _otel_trace
        from opentelemetry.sdk.trace import TracerProvider as _SDKTracerProvider

        if not isinstance(_otel_trace.get_tracer_provider(), _SDKTracerProvider):
            from opentelemetry.instrumentation.auto_instrumentation import initialize

            initialize()
    except Exception:
        pass  # SDK not installed — safe to continue without telemetry

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict
from uuid import uuid4

from dotenv import load_dotenv

# Load ~/.env so AzureOpenAI credentials and OTEL vars are available at import time.
load_dotenv(Path.home() / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from langchain.agents import create_agent as _create_react_agent  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import AzureChatOpenAI, ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import AnyMessage, add_messages  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ---------------------------------------------------------------------------
# LLM factory — auto-detects AzureOpenAI or standard OpenAI from env vars.
# Agent names have "gc" suffix to distinguish this gunicorn deployment.
# ---------------------------------------------------------------------------

_GC_SUFFIX = "_gc"


def _create_llm(
    agent_name: str, *, temperature: float, session_id: str
) -> ChatOpenAI | AzureChatOpenAI:
    """Create an LLM tagged with the gc-suffixed agent name."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    full_name = f"{agent_name}{_GC_SUFFIX}"
    tags = [f"agent:{full_name}", "travel-planner-gc"]
    metadata = {
        "agent_name": full_name,
        "agent_type": full_name,
        "session_id": session_id,
        "thread_id": session_id,
        "ls_model_name": model,
        "ls_temperature": temperature,
    }

    # Prefer OpenAI-compatible Azure endpoint (OPENAI_BASE_URL) — reports
    # gen_ai.request.model correctly in telemetry.
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
            temperature=temperature,
            tags=tags,
            metadata=metadata,
        )

    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        # AZURE_OPENAI_DEPLOYMENT is the canonical name used in Azure App Service docs;
        # AZURE_CHAT_DEPLOYMENT is the legacy name kept for backward compatibility.
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get(
            "AZURE_CHAT_DEPLOYMENT", model
        )
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_deployment=deployment,
            temperature=temperature,
            tags=tags,
            model_kwargs={"metadata": metadata},
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        tags=tags,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

DESTINATIONS = {
    "paris": {
        "country": "France",
        "currency": "EUR",
        "airport": "CDG",
        "highlights": [
            "Eiffel Tower at sunset",
            "Seine dinner cruise",
            "Day trip to Versailles",
        ],
    },
    "tokyo": {
        "country": "Japan",
        "currency": "JPY",
        "airport": "HND",
        "highlights": [
            "Tsukiji market food tour",
            "Ghibli Museum visit",
            "Day trip to Hakone hot springs",
        ],
    },
    "rome": {
        "country": "Italy",
        "currency": "EUR",
        "airport": "FCO",
        "highlights": [
            "Colosseum underground tour",
            "Private pasta masterclass",
            "Sunset walk through Trastevere",
        ],
    },
}


def _compute_dates() -> tuple[str, str]:
    start = datetime.now() + timedelta(days=30)
    end = start + timedelta(days=7)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def mock_search_flights(origin: str, destination: str, departure: str) -> str:
    """Return mock flight options for a given origin/destination pair."""
    random.seed(hash((origin, destination, departure)) % (2**32))
    airline = random.choice(["SkyLine", "AeroJet", "CloudNine"])
    fare = random.randint(700, 1250)
    return (
        f"Top choice: {airline} non-stop {origin}->{destination}, "
        f"depart {departure} 09:15, arrive 17:05. Premium economy ${fare} return."
    )


@tool
def mock_search_hotels(destination: str, check_in: str, check_out: str) -> str:
    """Return mock hotel recommendation for the stay."""
    random.seed(hash((destination, check_in, check_out)) % (2**32))
    name = random.choice(["Grand Meridian", "Hotel Lumière", "The Atlas"])
    rate = random.randint(240, 410)
    return (
        f"{name} near the historic centre. Boutique suites, rooftop bar, "
        f"average nightly rate ${rate} including breakfast."
    )


@tool
def mock_search_activities(destination: str) -> str:
    """Return a short list of signature activities for the destination."""
    data = DESTINATIONS.get(destination.lower(), DESTINATIONS["paris"])
    bullets = "\n".join(f"- {item}" for item in data["highlights"])
    return f"Signature experiences in {destination.title()}:\n{bullets}"


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class PlannerState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_request: str
    session_id: str
    origin: str
    destination: str
    departure: str
    return_date: str
    travellers: int
    flight_summary: Optional[str]
    hotel_summary: Optional[str]
    activities_summary: Optional[str]
    final_itinerary: Optional[str]
    current_agent: str


# ---------------------------------------------------------------------------
# LangGraph nodes — agent names use _gc suffix
# ---------------------------------------------------------------------------


def coordinator_node(state: PlannerState) -> PlannerState:
    llm = _create_llm("coordinator", temperature=0.2, session_id=state["session_id"])
    agent = _create_react_agent(llm, tools=[]).with_config(
        {
            "run_name": f"coordinator{_GC_SUFFIX}",
            "tags": ["agent", f"agent:coordinator{_GC_SUFFIX}"],
            "metadata": {
                "agent_name": f"coordinator{_GC_SUFFIX}",
                "session_id": state["session_id"],
            },
        }
    )
    system_message = SystemMessage(
        content=(
            "You are the lead travel coordinator. Extract the key details from the "
            "traveller's request and describe the plan for the specialist agents."
        )
    )
    result = agent.invoke({"messages": [system_message] + list(state["messages"])})
    final_message = result["messages"][-1]
    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "flight_specialist"
    return state


def flight_specialist_node(state: PlannerState) -> PlannerState:
    llm = _create_llm(
        "flight_specialist", temperature=0.4, session_id=state["session_id"]
    )
    agent = _create_react_agent(llm, tools=[mock_search_flights]).with_config(
        {
            "run_name": f"flight_specialist{_GC_SUFFIX}",
            "tags": ["agent", f"agent:flight_specialist{_GC_SUFFIX}"],
            "metadata": {
                "agent_name": f"flight_specialist{_GC_SUFFIX}",
                "session_id": state["session_id"],
            },
        }
    )
    step = (
        f"Find an appealing flight from {state['origin']} to {state['destination']} "
        f"departing {state['departure']} for {state['travellers']} travellers."
    )
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    state["flight_summary"] = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )
    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "hotel_specialist"
    return state


def hotel_specialist_node(state: PlannerState) -> PlannerState:
    llm = _create_llm(
        "hotel_specialist", temperature=0.5, session_id=state["session_id"]
    )
    agent = _create_react_agent(llm, tools=[mock_search_hotels]).with_config(
        {
            "run_name": f"hotel_specialist{_GC_SUFFIX}",
            "tags": ["agent", f"agent:hotel_specialist{_GC_SUFFIX}"],
            "metadata": {
                "agent_name": f"hotel_specialist{_GC_SUFFIX}",
                "session_id": state["session_id"],
            },
        }
    )
    step = (
        f"Recommend a boutique hotel in {state['destination']} between "
        f"{state['departure']} and {state['return_date']} for {state['travellers']} travellers."
    )
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    state["hotel_summary"] = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )
    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "activity_specialist"
    return state


def activity_specialist_node(state: PlannerState) -> PlannerState:
    llm = _create_llm(
        "activity_specialist", temperature=0.6, session_id=state["session_id"]
    )
    agent = _create_react_agent(llm, tools=[mock_search_activities]).with_config(
        {
            "run_name": f"activity_specialist{_GC_SUFFIX}",
            "tags": ["agent", f"agent:activity_specialist{_GC_SUFFIX}"],
            "metadata": {
                "agent_name": f"activity_specialist{_GC_SUFFIX}",
                "session_id": state["session_id"],
            },
        }
    )
    step = f"Curate signature activities for travellers spending a week in {state['destination']}."
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    state["activities_summary"] = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )
    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "plan_synthesizer"
    return state


def plan_synthesizer_node(state: PlannerState) -> PlannerState:
    llm = _create_llm(
        "plan_synthesizer", temperature=0.3, session_id=state["session_id"]
    )
    system_prompt = SystemMessage(
        content=(
            "You are the travel plan synthesiser. Combine the specialist insights into a "
            "concise, structured itinerary covering flights, accommodation and activities."
        )
    )
    content = json.dumps(
        {
            "flight": state["flight_summary"],
            "hotel": state["hotel_summary"],
            "activities": state["activities_summary"],
        },
        indent=2,
    )
    response = llm.invoke(
        [
            system_prompt,
            HumanMessage(
                content=(
                    f"Traveller request: {state['user_request']}\n\n"
                    f"Origin: {state['origin']} | Destination: {state['destination']}\n"
                    f"Dates: {state['departure']} to {state['return_date']}\n\n"
                    f"Specialist summaries:\n{content}"
                )
            ),
        ]
    )
    state["final_itinerary"] = response.content
    state["messages"].append(response)
    state["current_agent"] = "completed"
    return state


def should_continue(state: PlannerState) -> str:
    mapping = {
        "start": "coordinator",
        "flight_specialist": "flight_specialist",
        "hotel_specialist": "hotel_specialist",
        "activity_specialist": "activity_specialist",
        "plan_synthesizer": "plan_synthesizer",
    }
    return mapping.get(state["current_agent"], END)


def build_workflow() -> StateGraph:
    graph = StateGraph(PlannerState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("flight_specialist", flight_specialist_node)
    graph.add_node("hotel_specialist", hotel_specialist_node)
    graph.add_node("activity_specialist", activity_specialist_node)
    graph.add_node("plan_synthesizer", plan_synthesizer_node)
    graph.add_conditional_edges(START, should_continue)
    graph.add_conditional_edges("coordinator", should_continue)
    graph.add_conditional_edges("flight_specialist", should_continue)
    graph.add_conditional_edges("hotel_specialist", should_continue)
    graph.add_conditional_edges("activity_specialist", should_continue)
    graph.add_conditional_edges("plan_synthesizer", should_continue)
    return graph


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Travel Planner (Gunicorn)",
    description="LangGraph travel planner served by Gunicorn + Uvicorn workers with zero-code OTel.",
    version="0.1.0",
)


class PlanRequest(BaseModel):
    origin: str = "Seattle"
    destination: str = "Paris"
    travellers: int = 2
    user_request: Optional[str] = None


class PlanResponse(BaseModel):
    session_id: str
    origin: str
    destination: str
    departure: str
    return_date: str
    travellers: int
    flight_summary: Optional[str]
    hotel_summary: Optional[str]
    activities_summary: Optional[str]
    final_itinerary: Optional[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
async def plan(request: PlanRequest):
    """Run the multi-agent travel planner and return the itinerary."""
    session_id = str(uuid4())
    departure, return_date = _compute_dates()

    user_request = request.user_request or (
        f"Planning a week-long trip from {request.origin} to {request.destination}. "
        "Looking for a boutique hotel, comfortable flights and unique local experiences."
    )

    initial_state: PlannerState = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "session_id": session_id,
        "origin": request.origin,
        "destination": request.destination,
        "departure": departure,
        "return_date": return_date,
        "travellers": request.travellers,
        "flight_summary": None,
        "hotel_summary": None,
        "activities_summary": None,
        "final_itinerary": None,
        "current_agent": "start",
    }

    workflow = build_workflow()
    compiled = workflow.compile()
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 10}

    final_state: Optional[PlannerState] = None
    try:
        for step in compiled.stream(initial_state, config):
            _, node_state = next(iter(step.items()))
            final_state = node_state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not final_state:
        raise HTTPException(status_code=500, detail="Workflow produced no state")

    return PlanResponse(
        session_id=session_id,
        origin=request.origin,
        destination=request.destination,
        departure=departure,
        return_date=return_date,
        travellers=request.travellers,
        flight_summary=final_state.get("flight_summary"),
        hotel_summary=final_state.get("hotel_summary"),
        activities_summary=final_state.get("activities_summary"),
        final_itinerary=final_state.get("final_itinerary"),
    )
