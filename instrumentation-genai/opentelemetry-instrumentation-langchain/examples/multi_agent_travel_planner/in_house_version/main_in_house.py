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
Multi-agent travel planner driven by LangGraph.

The example coordinates a set of LangChain agents that collaborate to build a
week-long city break itinerary.  OpenTelemetry spans are produced for both the
overall orchestration and each LLM/tool call so you can inspect the traces in
your OTLP collector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from typing import cast
from uuid import uuid4
from http.server import HTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv
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


from langchain.agents import create_agent  # type: ignore[attr-defined]

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)

from opentelemetry.sdk._logs import LoggerProvider  # type: ignore[attr-defined]
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # type: ignore[attr-defined]
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# GenAI Utils imports - our in-house instrumentation framework
from opentelemetry.util.genai.handler import get_telemetry_handler, TelemetryHandler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    LLMInvocation,
    Workflow,
    InputMessage,
    OutputMessage,
    Text,
)

# ---------------------------------------------------------------------------
# Health check endpoint for Kubernetes
# ---------------------------------------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # Use HTTP/1.1 instead of HTTP/1.0

    def do_GET(self):
        if self.path == "/health":
            response_body = json.dumps({"status": "healthy"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        if self.path == "/plan":
            try:
                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode("utf-8"))

                # Extract parameters from request
                destination = request_data.get("destination", "Paris")
                budget = request_data.get("budget", 3000)
                duration = request_data.get("duration", 7)
                travelers = request_data.get("travelers", 2)
                interests = request_data.get("interests", ["sightseeing", "food"])

                # Create user request based on parameters
                user_request = (
                    f"We're planning a {duration}-day trip to {destination}. "
                    f"We have a budget of ${budget} and are interested in "
                    f"{', '.join(interests)}. We are {travelers} travelers."
                )

                # Generate session id here so workflow can be correlated
                session_id = str(uuid4())
                handler = get_telemetry_handler()

                # Run travel planner – it will instantiate and manage its own Workflow span.
                result = run_travel_planner(
                    session_id=session_id,
                    user_request=user_request,
                    destination_override=destination,
                    handler=handler,
                )

                # Allow asynchronous evaluations to finish before responding
                wait_seconds = int(os.getenv("EVAL_WAIT_SECONDS", "300"))
                if wait_seconds > 0:
                    print(
                        f"⏳ Waiting {wait_seconds}s for evaluations (session {session_id})"
                    )
                    time.sleep(wait_seconds)

                response_body = json.dumps(
                    {
                        "status": "success",
                        "session_id": session_id,
                        "itinerary": result.get(
                            "final_itinerary", "Planning completed"
                        ),
                        "request": user_request,
                    }
                ).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                import traceback

                traceback.print_exc()
                print(f"Error processing plan request: {e}")
                error_body = json.dumps({"status": "error", "message": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


def start_health_server():
    """Run the HTTP server that exposes health and planning endpoints."""
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise
    finally:
        server.server_close()


def run_travel_planner(
    *,
    session_id: str,
    user_request: str,
    destination_override: Optional[str] = None,
    handler: Optional[TelemetryHandler] = None,
) -> Dict[str, Any]:
    """Run the travel planner workflow and return results.

    Expects a started workflow span (child of HTTP SERVER span). Creates AgentInvocation
    and LLMInvocation spans as children. Populates messages for agents/LLM invocations.
    """
    # Instantiate workflow internally; do not append custom attributes.
    workflow = Workflow(
        name="travel_multi_agent_planner",
        workflow_type="graph",
        description="Multi-agent travel planner workflow",
        initial_input=user_request,
    )
    if not handler:
        handler = get_telemetry_handler()
    handler.start_workflow(workflow)

    origin = _pick_origin(user_request)
    destination = destination_override or _pick_destination(user_request)
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
        "telemetry_handler": handler,
    }

    graph = build_workflow()
    app = graph.compile()

    # RunnableConfig expects top-level keys; keep simple metadata
    config = {"configurable": {"thread_id": session_id}}

    print("🌍 Multi-Agent Travel Planner")
    print("=" * 60)

    final_state: Optional[PlannerState] = None

    for step in app.stream(initial_state, config):
        node_name, node_state = next(iter(step.items()))
        final_state = node_state
        print(f"\n🤖 {node_name.replace('_', ' ').title()} Agent")
        if node_state.get("messages"):
            last = node_state["messages"][-1]
            if isinstance(last, BaseMessage):
                content = last.content
                if isinstance(content, str):
                    preview = content[:400] + (
                        "... [truncated]" if len(content) > 400 else ""
                    )
                    print(preview)

    if not final_state:
        final_plan = ""
    else:
        final_plan = final_state.get("final_itinerary") or ""

    if final_plan:
        print("\n🎉 Final itinerary\n" + "-" * 40)
        print(final_plan)

    # Set final_output and stop the internally-created workflow (no extra attributes added).
    if final_plan:
        workflow.final_output = final_plan
    handler.stop_workflow(workflow)

    return {
        "session_id": session_id,
        "final_itinerary": final_plan,
        "flight_summary": final_state.get("flight_summary") if final_state else None,
        "hotel_summary": final_state.get("hotel_summary") if final_state else None,
        "activities_summary": final_state.get("activities_summary") if final_state else None,
    }


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


def _coerce_content(raw: Any) -> str:
    """Best-effort convert LangChain message content (which can be str | list | dict) into a string.

    LangChain tool / reasoning messages sometimes return structured lists of strings and dicts.
    For telemetry preview we flatten these deterministically while keeping JSON-like readability.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        # Join parts converting dicts to compact JSON
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            else:
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(repr(item))
        return "\n".join(parts)
    if isinstance(raw, dict):
        try:
            return json.dumps(raw, ensure_ascii=False)
        except Exception:
            return repr(raw)
    return str(raw)


# ---------------------------------------------------------------------------
# LLM metadata extraction helper
# ---------------------------------------------------------------------------

def _apply_llm_response_metadata(message: Any, llm_invocation: LLMInvocation) -> None:
    """Populate LLMInvocation from a LangChain response message.

    Responsibilities:
      1. If output_messages not already set, create a single assistant OutputMessage from message.content.
      2. Extract token usage from either:
         - message.usage_metadata => {input_tokens|prompt_tokens, output_tokens|completion_tokens}
         - message.response_metadata.token_usage OR .usage => {prompt_tokens, completion_tokens, total_tokens}
      3. Set llm_invocation.input_tokens / output_tokens when available.

    Safe to call multiple times; existing output_messages are preserved and not overwritten.
    Silently ignores missing metadata attributes.
    """
    if message is None:
        return

    # 1. Populate output_messages if absent.
    if not getattr(llm_invocation, "output_messages", None):
        raw_content = getattr(message, "content", None)
        # Some LangChain wrappers put text under .content; if absent, coerce entire message.
        coerced = _coerce_content(raw_content if raw_content is not None else message)
        llm_invocation.output_messages = [
            OutputMessage(
                role="assistant", parts=[Text(content=coerced)], finish_reason="stop"
            )
        ]

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    usage_meta: Any = getattr(message, "usage_metadata", None)
    if isinstance(usage_meta, dict) and usage_meta:
        in_val = usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens")
        out_val = usage_meta.get("output_tokens") or usage_meta.get("completion_tokens")
        if isinstance(in_val, int):
            input_tokens = in_val
        if isinstance(out_val, int):
            output_tokens = out_val

    if input_tokens is None or output_tokens is None:
        resp_meta: Any = getattr(message, "response_metadata", None)
        if isinstance(resp_meta, dict) and resp_meta:
            token_usage: Any = resp_meta.get("token_usage") or resp_meta.get("usage")
            if isinstance(token_usage, dict):
                if input_tokens is None:
                    prompt_val = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                    if isinstance(prompt_val, int):
                        input_tokens = prompt_val
                if output_tokens is None:
                    completion_val = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                    if isinstance(completion_val, int):
                        output_tokens = completion_val

    if input_tokens is not None:
        llm_invocation.input_tokens = input_tokens
    if output_tokens is not None:
        llm_invocation.output_tokens = output_tokens


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
    # Telemetry context
    telemetry_handler: Any


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5-nano")


def _create_llm(agent_name: str, *, temperature: float, session_id: str) -> ChatOpenAI:
    """Create an LLM instance decorated with tags/metadata for tracing."""
    model = _model_name()
    tags = [f"agent:{agent_name}", "travel-planner"]
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


def _configure_manual_instrumentation() -> None:
    """Initialise trace and metric providers that export to the configured OTLP endpoint."""
    """Configure tracing/metrics/logging manually once per process so exported data goes to OTLP."""

    from opentelemetry import _events, _logs, trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._events import EventLoggerProvider
    
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )
    _events.set_event_logger_provider(EventLoggerProvider())

# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def coordinator_node(state: PlannerState) -> PlannerState:
    handler: TelemetryHandler = state["telemetry_handler"]
    agent = AgentInvocation(
        name="coordinator",
        agent_type="coordinator",
        model=_model_name(),
        system_instructions=(
            "You are the lead travel coordinator. Extract key details from the user's request, "
            "outline required specialist agents (flight, hotel, activities, synthesis) and provide planning guidance."
        ),
        input_context=state.get("user_request"),
    )
    handler.start_agent(agent)

    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        operation="chat",
        input_messages=[
            InputMessage(role="system", parts=[Text(content="Lead coordinator")]),
            InputMessage(
                role="user", parts=[Text(content=state.get("user_request", ""))]
            ),
        ],
    )
    llm_invocation.provider = "openai"
    llm_invocation.framework = "langgraph"
    handler.start_llm(llm_invocation)

    llm = _create_llm("coordinator", temperature=0.2, session_id=state["session_id"])
    system_message = SystemMessage(
        content=(
            "You are the lead travel coordinator. Extract the key details from the "
            "traveller's request and describe the plan for the specialist agents."
        )
    )
    response = llm.invoke([system_message] + state["messages"])
    response_text = _coerce_content(response.content)

    # Populate output messages + token usage via helper.
    _apply_llm_response_metadata(response, llm_invocation)
    handler.stop_llm(llm_invocation)

    agent.output_result = response_text
    state["messages"].append(cast(AnyMessage, response))
    state["current_agent"] = "flight_specialist"
    handler.stop_agent(agent)
    return state


def flight_specialist_node(state: PlannerState) -> PlannerState:
    handler: TelemetryHandler = state["telemetry_handler"]
    step_input = (
        f"Find an appealing flight from {state['origin']} to {state['destination']} "
        f"departing {state['departure']} for {state['travellers']} travellers."
    )
    agent_invocation = AgentInvocation(
        name="flight_specialist",
        agent_type="specialist",
        model=_model_name(),
        tools=["mock_search_flights"],
        system_instructions=(
            "You are a flight specialist. Given origin, destination, dates and traveller count, "
            "recommend the most appealing flight option with concise justification."
        ),
    )
    handler.start_agent(agent_invocation)
    
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        operation="chat",
        input_messages=[
            InputMessage(
                role="user",
                parts=[Text(content=step_input)],
            )
        ],
    )
    llm_invocation.provider = "openai"
    llm_invocation.framework = "langgraph"
    handler.start_llm(llm_invocation)

    llm = _create_llm(
        "flight_specialist", temperature=0.4, session_id=state["session_id"]
    )
    react_agent = create_agent(llm, tools=[mock_search_flights]).with_config(
        {
            "run_name": "flight_specialist",
            "tags": ["agent", "agent:flight_specialist"],
            "metadata": {
                "agent_name": "flight_specialist",
                "agent_type": "specialist",
                "session_id": state["session_id"],
            },
        }
    )
    step = (
        f"Find an appealing flight from {state['origin']} to {state['destination']} "
        f"departing {state['departure']} for {state['travellers']} travellers."
    )
    result = react_agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    summary = _coerce_content(
        final_message.content
        if isinstance(final_message, BaseMessage)
        else final_message
    )
    state["flight_summary"] = summary

    _apply_llm_response_metadata(final_message, llm_invocation)
    handler.stop_llm(llm_invocation)

    agent_invocation.output_result = summary
    state["messages"].append(
        cast(
            AnyMessage,
            final_message
            if isinstance(final_message, BaseMessage)
            else AIMessage(content=summary),
        )
    )
    state["current_agent"] = "hotel_specialist"
        # Attach agent output messages
    # No direct output messages attribute assignment; output_result contains summary

    handler.stop_agent(agent_invocation)
    return state


def hotel_specialist_node(state: PlannerState) -> PlannerState:
    handler: TelemetryHandler = state["telemetry_handler"]
    step_input = (
        f"Recommend hotels in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    agent_invocation = AgentInvocation(
        name="hotel_specialist",
        agent_type="specialist",
        model=_model_name(),
        tools=["mock_search_hotels"],
        system_instructions=(
            "You are a hotel specialist. Recommend a single standout hotel option matching the trip context, "
            "highlighting style, location and approximate nightly rate."
        ),
    )
    handler.start_agent(agent_invocation)

    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        operation="chat",
        input_messages=[
            InputMessage(
                role="user",
                parts=[Text(content=step_input)],
            )
        ],
    )
    llm_invocation.provider = "openai"
    llm_invocation.framework = "langgraph"
    handler.start_llm(llm_invocation)

    llm = _create_llm(
        "hotel_specialist", temperature=0.3, session_id=state["session_id"]
    )
    react_agent = create_agent(llm, tools=[mock_search_hotels]).with_config(
        {
            "run_name": "hotel_specialist",
            "tags": ["agent", "agent:hotel_specialist"],
            "metadata": {
                "agent_name": "hotel_specialist",
                "agent_type": "specialist",
                "session_id": state["session_id"],
            },
        }
    )
    step = (
        f"Recommend hotels in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    result = react_agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    summary = _coerce_content(
        final_message.content
        if isinstance(final_message, BaseMessage)
        else final_message
    )
    state["hotel_summary"] = summary

    _apply_llm_response_metadata(final_message, llm_invocation)
    handler.stop_llm(llm_invocation)

    agent_invocation.output_result = summary
    state["messages"].append(
        cast(
            AnyMessage,
            final_message
            if isinstance(final_message, BaseMessage)
            else AIMessage(content=summary),
        )
    )
    state["current_agent"] = "activity_specialist"
    # No direct output messages attribute assignment; output_result contains summary
    handler.stop_agent(agent_invocation)
    return state


def activity_specialist_node(state: PlannerState) -> PlannerState:
    handler: TelemetryHandler = state["telemetry_handler"]
    step_input = (
        f"Recommend activities in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    agent_invocation = AgentInvocation(
        name="activity_specialist",
        agent_type="specialist",
        model=_model_name(),
        tools=["mock_search_activities"],
        system_instructions=(
            "You are an activity specialist. Suggest signature destination activities that fit the trip goals and dates."
        ),
    )
    handler.start_agent(agent_invocation)

    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        operation="chat",
        input_messages=[
            InputMessage(
                role="user",
                parts=[Text(content=step_input)],
            )
        ],
    )
    llm_invocation.provider = "openai"
    llm_invocation.framework = "langgraph"
    handler.start_llm(llm_invocation)

    llm = _create_llm(
        "activity_specialist", temperature=0.5, session_id=state["session_id"]
    )
    react_agent = create_agent(llm, tools=[mock_search_activities]).with_config(
        {
            "run_name": "activity_specialist",
            "tags": ["agent", "agent:activity_specialist"],
            "metadata": {
                "agent_name": "activity_specialist",
                "agent_type": "specialist",
                "session_id": state["session_id"],
            },
        }
    )
    step = (
        f"Recommend activities in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    result = react_agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    summary = _coerce_content(
        final_message.content
        if isinstance(final_message, BaseMessage)
        else final_message
    )
    # Store under activities_summary key (TypedDict field)
    state["activities_summary"] = summary

    _apply_llm_response_metadata(final_message, llm_invocation)
    handler.stop_llm(llm_invocation)

    agent_invocation.output_result = summary
    state["messages"].append(
        cast(
            AnyMessage,
            final_message
            if isinstance(final_message, BaseMessage)
            else AIMessage(content=summary),
        )
    )
    state["current_agent"] = "plan_synthesizer"
    # No direct output messages attribute assignment; output_result contains summary
    handler.stop_agent(agent_invocation)
    return state


def plan_synthesizer_node(state: PlannerState) -> PlannerState:
    handler: TelemetryHandler = state["telemetry_handler"]
    agent_invocation = AgentInvocation(
        name="plan_synthesizer",
        agent_type="synthesizer",
        model=_model_name(),
        system_instructions=(
            "You synthesize the specialist outputs into a cohesive, structured itinerary (flights, hotel, activities)."
        ),
    )
    handler.start_agent(agent_invocation)

    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        operation="chat",
        input_messages=[
            InputMessage(role="system", parts=[Text(content="Plan synthesiser")]),
            InputMessage(
                role="user",
                parts=[Text(content="Combine specialist insights into itinerary")],
            ),
        ],
    )
    llm_invocation.provider = "openai"
    llm_invocation.framework = "langgraph"
    handler.start_llm(llm_invocation)

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
    response_text = _coerce_content(response.content)
    state["final_itinerary"] = _coerce_content(response.content)
    state["messages"].append(cast(AnyMessage, response))
    state["current_agent"] = "completed"

    _apply_llm_response_metadata(response, llm_invocation)
    handler.stop_llm(llm_invocation)

    agent_invocation.output_result = response_text
    # No direct output messages attribute assignment; output_result contains summary
    handler.stop_agent(agent_invocation)
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


def build_workflow() -> StateGraph:  # type: ignore[return-type]
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


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options for running the planner."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-agent travel planner demo. Run as a long-lived service "
            "(default) or execute a single planning session with --once."
        )
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one planning session locally, print the itinerary, then exit.",
    )
    parser.add_argument(
        "--request",
        type=str,
        help="Override the default travel request when using --once.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_cli_args(argv)
    # Load environment variables from .env file
    load_dotenv()

    _configure_manual_instrumentation()
    # Using GenAI Utils manual instrumentation (in-house framework)
    # To compare with auto-instrumentation, you can enable this instead:
    # LangchainInstrumentor().instrument()

    print("🌍 Travel Planner Service Starting...")
    print("🏥 Health check endpoint: http://localhost:8080/health")
    print("📋 Planning endpoint: http://localhost:8080/plan")

    try:
        if args.once:
            user_request = args.request or (
                "We're planning a romantic long-week trip to Paris from Seattle "
                "next month. We'd love a boutique hotel, business-class flights "
                "and a few unique experiences."
            )
            print(f"🚀 Running single request: {user_request}")
            result = run_travel_planner(user_request=user_request, session_id="fake-session-001")
            print(f"✅ Planning completed for session: {result['session_id']}")

            wait_seconds = int(os.getenv("EVAL_WAIT_SECONDS", "300"))
            if wait_seconds > 0:
                print(
                    f"\n⏳ Waiting {wait_seconds} seconds for evaluations to finish..."
                )
                time.sleep(wait_seconds)
                print("✅ Evaluation wait complete, shutting down...")
        else:
            print("🚀 Starting as HTTP service...")
            start_health_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received. Cleaning up...")
    finally:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()


if __name__ == "__main__":
    main()
