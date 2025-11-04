#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-agent travel planner using Traceloop SDK with zero-code translator.

This version uses Traceloop SDK decorators (@workflow, @task) and relies on the
Traceloop translator to automatically convert traceloop.* attributes to gen_ai.*
semantic conventions via zero-code instrumentation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional, TypedDict
from uuid import uuid4

# Configure Python logging to DEBUG level to see our trace messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Enable debug logging for specific modules
logging.getLogger('opentelemetry.util.genai.processor.traceloop_span_processor').setLevel(logging.DEBUG)
logging.getLogger('opentelemetry.util.genai.handler').setLevel(logging.DEBUG)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

try:  # LangChain >= 1.0.0
    from langchain.agents import (
        create_agent as _create_react_agent,  # type: ignore[attr-defined]
    )
except (
    ImportError
):  # pragma: no cover - compatibility with older LangGraph releases
    from langgraph.prebuilt import (
        create_react_agent as _create_react_agent,  # type: ignore[assignment]
    )

# Import Traceloop SDK
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# Import OpenTelemetry components for logging
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

# Get configuration from environment variables
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "travel-planner-traceloop")
OTEL_RESOURCE_ATTRIBUTES = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is required", file=sys.stderr)
    sys.exit(1)

# Convert gRPC endpoint (port 4317) to HTTP endpoint (port 4318) for Traceloop
# Note: Kubernetes will expand $(SPLUNK_OTEL_AGENT) automatically in the YAML
if ":4317" in OTEL_EXPORTER_OTLP_ENDPOINT:
    OTEL_EXPORTER_OTLP_ENDPOINT = OTEL_EXPORTER_OTLP_ENDPOINT.replace(":4317", ":4318")
    print(f"Note: Converted gRPC endpoint to HTTP endpoint for Traceloop: {OTEL_EXPORTER_OTLP_ENDPOINT}")

print(f"Service Name: {OTEL_SERVICE_NAME}")
print(f"OTLP Endpoint: {OTEL_EXPORTER_OTLP_ENDPOINT}")
print(f"Resource Attributes: {OTEL_RESOURCE_ATTRIBUTES}")

# Parse resource attributes
resource_attributes = {}
if OTEL_RESOURCE_ATTRIBUTES:
    for attr in OTEL_RESOURCE_ATTRIBUTES.split(","):
        if "=" in attr:
            key, value = attr.split("=", 1)
            resource_attributes[key.strip()] = value.strip()

# Initialize Traceloop SDK
# The Traceloop translator will automatically convert traceloop.* to gen_ai.* attributes
Traceloop.init(
    disable_batch=True,
    api_endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
    app_name=OTEL_SERVICE_NAME,
    resource_attributes=resource_attributes
)
print("✓ Traceloop SDK initialized with zero-code translator")


def _configure_otlp_logging() -> None:
    """
    Initialize a logger provider that exports to the configured OTLP endpoint.
    
    This is needed for evaluation results to be emitted as OTLP log records.
    Traceloop SDK handles traces, but we need to explicitly configure logs.
    """
    from opentelemetry._logs import get_logger_provider
    
    # Check if already configured
    try:
        existing = get_logger_provider()
        if isinstance(existing, LoggerProvider):
            print("✓ LoggerProvider already configured")
            return
    except:
        pass
    
    # Parse resource attributes from environment (same as Traceloop)
    resource_attrs = {"service.name": OTEL_SERVICE_NAME}
    if OTEL_RESOURCE_ATTRIBUTES:
        for attr in OTEL_RESOURCE_ATTRIBUTES.split(","):
            if "=" in attr:
                key, value = attr.split("=", 1)
                resource_attrs[key.strip()] = value.strip()
    
    resource = Resource(attributes=resource_attrs)
    logger_provider = LoggerProvider(resource=resource)
    
    # Use HTTP exporter since Traceloop uses HTTP/protobuf (port 4318)
    # HTTP OTLP exporter needs the full path including /v1/logs
    log_endpoint = OTEL_EXPORTER_OTLP_ENDPOINT
    if not log_endpoint.endswith("/v1/logs"):
        log_endpoint = f"{log_endpoint.rstrip('/')}/v1/logs"
    
    log_processor = BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=log_endpoint)
    )
    logger_provider.add_log_record_processor(log_processor)
    set_logger_provider(logger_provider)
    print(f"✓ OTLP logging configured with endpoint: {log_endpoint}")


# Configure logging for evaluation results
_configure_otlp_logging()

# ---------------------------------------------------------------------------
# Single-Library Solution: Message Reconstruction in Translator
# ---------------------------------------------------------------------------
# NEW APPROACH: The Traceloop translator now reconstructs LangChain message objects
# directly from Traceloop's serialized JSON data (traceloop.entity.input/output).
# 
# This eliminates the need for LangChain instrumentation!
#
# How it works:
# 1. Traceloop SDK creates spans with traceloop.entity.input/output (JSON strings)
# 2. TraceloopSpanProcessor extracts and parses the JSON
# 3. Reconstructs HumanMessage, AIMessage, etc. objects
# 4. Sets them on LLMInvocation.input_messages/output_messages
# 5. Evaluators receive full message objects → evaluations work!
#
# Benefits:
# - Single library (Traceloop SDK only, no dual instrumentation)
# - No circular import issues (different initialization path)
# - Simpler architecture (one instrumentation instead of two)
# - Better performance (one callback instead of two)
#
# Note: langchain-core must be installed for message reconstruction to work,
# but LangChain instrumentation is NOT needed.
print("✓ Message reconstruction enabled in translator (no LangChain instrumentation needed)")

# ---------------------------------------------------------------------------
# Sample data utilities
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


def _pick_destination(user_request: str) -> str:
    lowered = user_request.lower()
    for name in DESTINATIONS:
        if name in lowered:
            return name.title()
    return "Paris"


def _pick_origin(user_request: str) -> str:
    lowered = user_request.lower()
    for city in ["seattle", "new york", "san francisco", "london"]:
        if city in lowered:
            return city.title()
    return "Seattle"


def _compute_dates() -> tuple[str, str]:
    start = datetime.now() + timedelta(days=30)
    end = start + timedelta(days=7)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tools exposed to agents
# ---------------------------------------------------------------------------


@tool
def mock_search_flights(origin: str, destination: str, departure: str) -> str:
    """Return mock flight options for a given origin/destination pair."""
    random.seed(hash((origin, destination, departure)) % (2**32))
    airline = random.choice(["SkyLine", "AeroJet", "CloudNine"])
    fare = random.randint(700, 1250)
    return (
        f"Top choice: {airline} non-stop service {origin}->{destination}, "
        f"depart {departure} 09:15, arrive {departure} 17:05. "
        f"Premium economy fare ${fare} return."
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
# LangGraph state & helpers
# ---------------------------------------------------------------------------


class PlannerState(TypedDict):
    """Shared state that moves through the LangGraph workflow."""

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


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _create_llm(
    agent_name: str, *, temperature: float, session_id: str
) -> ChatOpenAI:
    """Create an LLM instance decorated with tags/metadata for tracing."""
    model = _model_name()
    tags = [f"agent:{agent_name}", "travel-planner-traceloop"]
    metadata = {
        "agent_name": agent_name,
        "agent_type": agent_name,
        "session_id": session_id,
        "thread_id": session_id,
        "ls_model_name": model,
        "ls_temperature": temperature,
    }
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        tags=tags,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# LangGraph nodes with Traceloop @task decorators
# ---------------------------------------------------------------------------


@task(name="coordinator_agent")
def coordinator_node(state: PlannerState) -> PlannerState:
    """Coordinate the travel planning workflow."""
    llm = _create_llm(
        "coordinator", temperature=0.2, session_id=state["session_id"]
    )
    system_message = SystemMessage(
        content=(
            "You are the lead travel coordinator. Extract the key details from the "
            "traveller's request and describe the plan for the specialist agents."
        )
    )
    response = llm.invoke([system_message] + state["messages"])

    state["messages"].append(response)
    state["current_agent"] = "flight_specialist"
    return state


@task(name="flight_specialist_agent")
def flight_specialist_node(state: PlannerState) -> PlannerState:
    """Search and recommend flights."""
    llm = _create_llm(
        "flight_specialist", temperature=0.4, session_id=state["session_id"]
    )
    agent = (
        _create_react_agent(llm, tools=[mock_search_flights]).with_config(
            {
                "run_name": "flight_specialist",
                "tags": ["agent", "agent:flight_specialist"],
                "metadata": {
                    "agent_name": "flight_specialist",
                    "session_id": state["session_id"],
                },
            }
        )
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


@task(name="hotel_specialist_agent")
def hotel_specialist_node(state: PlannerState) -> PlannerState:
    """Search and recommend hotels."""
    llm = _create_llm(
        "hotel_specialist", temperature=0.5, session_id=state["session_id"]
    )
    agent = (
        _create_react_agent(llm, tools=[mock_search_hotels]).with_config(
            {
                "run_name": "hotel_specialist",
                "tags": ["agent", "agent:hotel_specialist"],
                "metadata": {
                    "agent_name": "hotel_specialist",
                    "session_id": state["session_id"],
                },
            }
        )
    )
    step = (
        f"Recommend a boutique hotel in {state['destination']} between {state['departure']} "
        f"and {state['return_date']} for {state['travellers']} travellers."
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


@task(name="activity_specialist_agent")
def activity_specialist_node(state: PlannerState) -> PlannerState:
    """Search and recommend activities."""
    llm = _create_llm(
        "activity_specialist", temperature=0.6, session_id=state["session_id"]
    )
    agent = (
        _create_react_agent(llm, tools=[mock_search_activities]).with_config(
            {
                "run_name": "activity_specialist",
                "tags": ["agent", "agent:activity_specialist"],
                "metadata": {
                    "agent_name": "activity_specialist",
                    "session_id": state["session_id"],
                },
            }
        )
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


@task(name="plan_synthesizer_agent")
def plan_synthesizer_node(state: PlannerState) -> PlannerState:
    """Synthesize all recommendations into a final itinerary."""
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
# Entry point with @workflow decorator
# ---------------------------------------------------------------------------


@workflow(name="travel_planner_multi_agent")
def main() -> None:
    """Main workflow for multi-agent travel planning."""
    session_id = str(uuid4())
    user_request = (
        "We're planning a romantic long-week trip to Paris from Seattle next month. "
        "We'd love a boutique hotel, business-class flights and a few unique experiences."
    )

    origin = _pick_origin(user_request)
    destination = _pick_destination(user_request)
    departure, return_date = _compute_dates()

    initial_state: PlannerState = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,
        "session_id": session_id,
        "origin": origin,
        "destination": destination,
        "departure": departure,
        "return_date": return_date,
        "travellers": 2,
        "flight_summary": None,
        "hotel_summary": None,
        "activities_summary": None,
        "final_itinerary": None,
        "current_agent": "start",
    }

    workflow = build_workflow()
    app = workflow.compile()

    print("🌍 Multi-Agent Travel Planner (Traceloop SDK)")
    print("=" * 60)

    final_state: Optional[PlannerState] = None

    for step in app.stream(initial_state, {"configurable": {"thread_id": session_id}, "recursion_limit": 10}):
        node_name, node_state = next(iter(step.items()))
        final_state = node_state
        print(f"\n🤖 {node_name.replace('_', ' ').title()} Agent")
        if node_state.get("messages"):
            last = node_state["messages"][-1]
            if isinstance(last, BaseMessage):
                preview = last.content
                if len(preview) > 400:
                    preview = preview[:400] + "... [truncated]"
                print(preview)

    if not final_state:
        final_plan = ""
    else:
        final_plan = final_state.get("final_itinerary") or ""

    if final_plan:
        print("\n🎉 Final itinerary\n" + "-" * 40)
        print(final_plan)


def flush_telemetry():
    """Flush all OpenTelemetry providers before exit."""
    print("\n🔄 Flushing telemetry data...", flush=True)
    
    # Flush traces (Traceloop SDK uses OTel TracerProvider under the hood)
    try:
        from opentelemetry import trace
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            print("  → Flushing traces (traceloop.* and gen_ai.* spans)...", flush=True)
            tracer_provider.force_flush(timeout_millis=30000)  # 30 seconds
    except Exception as e:
        print(f"  ⚠️  Could not flush traces: {e}", flush=True)
    
    # Flush logs (if any emitters are using logs)
    try:
        from opentelemetry._logs import get_logger_provider
        logger_provider = get_logger_provider()
        if hasattr(logger_provider, "force_flush"):
            print("  → Flushing logs...", flush=True)
            logger_provider.force_flush(timeout_millis=30000)  # 30 seconds
    except Exception as e:
        print(f"  ⚠️  Could not flush logs: {e}", flush=True)
    
    # Flush metrics
    try:
        from opentelemetry.metrics import get_meter_provider
        meter_provider = get_meter_provider()
        if hasattr(meter_provider, "force_flush"):
            print("  → Flushing metrics...", flush=True)
            meter_provider.force_flush(timeout_millis=30000)  # 30 seconds
    except Exception as e:
        print(f"  ⚠️  Could not flush metrics: {e}", flush=True)
    
    print("✅ Telemetry flush complete!\n", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
        print("\n✓ Workflow completed successfully!")
        print("✓ Traces exported with traceloop.* attributes")
        print("✓ Zero-code translator converted to gen_ai.* attributes")
    except Exception as e:
        print(f"\n✗ ERROR: Workflow failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # ALWAYS flush telemetry, even on errors
        # This ensures both traceloop.* and translated gen_ai.* spans are exported
        flush_telemetry()
        sys.exit(exit_code)

