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

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
# from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.trace import SpanKind
from opentelemetry.semconv.resource import ResourceAttributes

import sys

# GenAI Utils imports - our in-house instrumentation framework
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    AgentInvocation,
    LLMInvocation,
    ToolCall,
)

# LangChain callback imports for token extraction
from langchain_core.callbacks import BaseCallbackHandler


# ---------------------------------------------------------------------------
# Telemetry callback for token and tool tracking
# ---------------------------------------------------------------------------


class TelemetryCallback(BaseCallbackHandler):
    """Callback to capture token usage, messages, and tool calls from LangChain."""

    def __init__(self, handler):
        self.tokens = {}
        self.handler = handler
        self.tool_calls = {}  # Map run_id to ToolCall object
        self.input_messages = []
        self.output_messages = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Capture input messages."""
        from opentelemetry.util.genai.types import InputMessage, Text
        
        # Capture input messages (prompts)
        self.input_messages = []
        if prompts:
            for prompt in prompts:
                self.input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[Text(content=prompt)]
                    )
                )

    def on_llm_end(self, response, **kwargs):
        """Capture token usage and output messages from LLM response."""
        from opentelemetry.util.genai.types import OutputMessage, Text
        
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.tokens = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }
        
        # Capture output messages
        self.output_messages = []
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    self.output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=gen.text)],
                            finish_reason="stop"
                        )
                    )

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Capture tool call start and create ToolCall span."""
        run_id = kwargs.get("run_id")
        
        # Extract tool name and ID from serialized data
        payload = serialized or {}
        tool_name = payload.get("name") or payload.get("id") or kwargs.get("name", "unknown_tool")
        
        # Extract tool ID (similar to auto-instrumentation)
        id_source = payload.get("id") or kwargs.get("id")
        if isinstance(id_source, (list, tuple)):
            tool_id = ".".join(str(part) for part in id_source)
        elif id_source is not None:
            tool_id = str(id_source)
        else:
            tool_id = None
        
        # Create ToolCall span at the right timing (during LLM context)
        tool_call = ToolCall(
            name=tool_name,
            id=tool_id,
            arguments=input_str,
            framework="langgraph",
        )
        
        if self.handler:
            self.handler.start_tool_call(tool_call)
            self.tool_calls[run_id] = tool_call

    def on_tool_end(self, output, **kwargs):
        """Capture tool call end and stop ToolCall span."""
        run_id = kwargs.get("run_id")
        tool_call = self.tool_calls.get(run_id)
        
        if tool_call and self.handler:
            self.handler.stop_tool_call(tool_call)
            del self.tool_calls[run_id]

    def on_tool_error(self, error, **kwargs):
        """Capture tool call error."""
        run_id = kwargs.get("run_id")
        tool_call = self.tool_calls.get(run_id)
        
        if tool_call and self.handler:
            from opentelemetry.util.genai.types import Error as GenAIError
            self.handler.fail_tool_call(tool_call, GenAIError(message=str(error), type=type(error).__name__))
            del self.tool_calls[run_id]


# ---------------------------------------------------------------------------
# Health check endpoint for Kubernetes
# ---------------------------------------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # Use HTTP/1.1 instead of HTTP/1.0
    
    def do_GET(self):
        if self.path == '/health':
            response_body = json.dumps({"status": "healthy"}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/plan':
            try:
                # Read request body
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                
                # Extract parameters from request
                destination = request_data.get('destination', 'Paris')
                budget = request_data.get('budget', 3000)
                duration = request_data.get('duration', 7)
                travelers = request_data.get('travelers', 2)
                interests = request_data.get('interests', ['sightseeing', 'food'])
                
                # Create user request based on parameters
                user_request = (
                    f"We're planning a {duration}-day trip to {destination}. "
                    f"We have a budget of ${budget} and are interested in "
                    f"{', '.join(interests)}. We are {travelers} travelers."
                )
                
                # Run the travel planner
                result = run_travel_planner(user_request, destination)

                # Allow asynchronous evaluations to finish before responding
                wait_seconds = int(os.getenv("EVAL_WAIT_SECONDS", "150"))
                if wait_seconds > 0:
                    print(
                        f"⏳ Waiting {wait_seconds}s for evaluations (session {result.get('session_id')})"
                    )
                    time.sleep(wait_seconds)

                response_body = json.dumps({
                    "status": "success",
                    "session_id": result.get("session_id"),
                    "itinerary": result.get("final_itinerary", "Planning completed"),
                    "request": user_request
                }).encode()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing plan request: {e}")
                error_body = json.dumps({
                    "status": "error",
                    "message": str(e)
                }).encode()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()

def start_health_server():
    """Run the HTTP server that exposes health and planning endpoints."""
    server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise
    finally:
        server.server_close()


def run_travel_planner(user_request: str, destination_override: Optional[str] = None) -> Dict[str, Any]:
    """Run the travel planner workflow and return results"""
    session_id = str(uuid4())
    
    origin = _pick_origin(user_request)
    destination = destination_override or _pick_destination(user_request)
    departure, return_date = _compute_dates()

    # Initialize telemetry handler for manual instrumentation
    handler = get_telemetry_handler()

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

    workflow = build_workflow()
    app = workflow.compile()

    tracer = trace.get_tracer(__name__)
    attributes = _trace_attributes_for_root(initial_state)
    root_input = [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": user_request,
                }
            ],
        }
    ]

    with tracer.start_as_current_span(
        name="invoke_agent travel_multi_agent_planner",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as root_span:
        root_span.set_attribute(
            "gen_ai.input.messages", json.dumps(root_input)
        )

        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 10,
        }

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

        root_span.set_attribute("gen_ai.response.model", _model_name())
        if final_plan:
            root_span.set_attribute(
                "gen_ai.output.messages",
                json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {"type": "text", "content": final_plan[:4000]}
                            ],
                            "finish_reason": "stop",
                        }
                    ]
                ),
            )
            preview = final_plan[:500] + (
                "..." if len(final_plan) > 500 else ""
            )
            root_span.set_attribute("metadata.final_plan.preview", preview)
        root_span.set_attribute("metadata.session_id", session_id)
        root_span.set_attribute(
            "metadata.agents_used",
            len(
                [
                    key
                    for key in [
                        "flight_summary",
                        "hotel_summary",
                        "activities_summary",
                    ]
                    if final_state and final_state.get(key)
                ]
            ),
        )

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
    return os.getenv("OPENAI_MODEL", "gpt-4.1")


def _create_llm(
    agent_name: str, *, temperature: float, session_id: str
) -> ChatOpenAI:
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


def _configure_otlp_tracing() -> None:
    """Initialise trace and metric providers that export to the configured OTLP endpoint."""
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    
    # Create resource with service name from environment
    service_name = os.getenv("OTEL_SERVICE_NAME", "unknown_service")
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name
    })
    
    # Configure trace provider
    tracer_provider = TracerProvider(resource=resource)
    trace_processor = BatchSpanProcessor(OTLPSpanExporter())
    tracer_provider.add_span_processor(trace_processor)
    trace.set_tracer_provider(tracer_provider)
    
    # Configure metrics provider
    metric_exporter = OTLPMetricExporter()
    metric_reader = PeriodicExportingMetricReader(
        exporter=metric_exporter,
        export_interval_millis=5000  # Export every 5 seconds
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(meter_provider)


def _trace_attributes_for_root(state: PlannerState) -> Dict[str, str]:
    """Attributes attached to the root GenAI span."""
    provider_name = "openai"
    server_address = os.getenv("OPENAI_BASE_URL", "api.openai.com")
    return {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": provider_name,
        "gen_ai.request.model": _model_name(),
        "gen_ai.agent.name": "travel_multi_agent_planner",
        "gen_ai.agent.id": f"travel_planner_{state['session_id']}",
        "gen_ai.conversation.id": state["session_id"],
        "gen_ai.request.temperature": 0.4,
        "gen_ai.request.top_p": 1.0,
        "gen_ai.request.max_tokens": 1024,
        "gen_ai.request.frequency_penalty": 0.0,
        "gen_ai.request.presence_penalty": 0.0,
        "server.address": server_address.replace("https://", "")
        .replace("http://", "")
        .rstrip("/"),
        "server.port": "443",
        "service.name": os.getenv(
            "OTEL_SERVICE_NAME",
            "opentelemetry-python-langchain-multi-agent-in-house",
        ),
    }


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def coordinator_node(state: PlannerState) -> PlannerState:
    handler = state.get("telemetry_handler")

    # Create and start AgentInvocation
    agent = AgentInvocation(
        name="coordinator",
        agent_type="coordinator",
        framework="langgraph",
        model=_model_name(),
    )
    if handler:
        handler.start_agent(agent)

    # Create callback for token tracking
    callback = TelemetryCallback(handler)

    # Create and start LLMInvocation
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        provider="openai",
        framework="langgraph",
        operation="chat",
    )
    if handler:
        handler.start_llm(llm_invocation)

    llm = _create_llm(
        "coordinator", temperature=0.2, session_id=state["session_id"]
    )
    system_message = SystemMessage(
        content=(
            "You are the lead travel coordinator. Extract the key details from the "
            "traveller's request and describe the plan for the specialist agents."
        )
    )
    response = llm.invoke(
        [system_message] + state["messages"], config={"callbacks": [callback]}
    )

    # Update LLMInvocation with token counts and messages
    llm_invocation.input_tokens = callback.tokens.get("input", 0)
    llm_invocation.output_tokens = callback.tokens.get("output", 0)
    llm_invocation.input_messages = callback.input_messages
    llm_invocation.output_messages = callback.output_messages

    # Stop LLMInvocation
    if handler:
        handler.stop_llm(llm_invocation)

    state["messages"].append(response)
    state["current_agent"] = "flight_specialist"

    # Stop AgentInvocation
    if handler:
        handler.stop_agent(agent)

    return state


def flight_specialist_node(state: PlannerState) -> PlannerState:
    handler = state.get("telemetry_handler")

    # Create and start AgentInvocation
    agent_invocation = AgentInvocation(
        name="flight_specialist",
        agent_type="specialist",
        framework="langgraph",
        model=_model_name(),
        tools=["mock_search_flights"],
    )
    if handler:
        handler.start_agent(agent_invocation)

    # Create callback for token and tool tracking
    callback = TelemetryCallback(handler)

    # Create and start LLMInvocation
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        provider="openai",
        framework="langgraph",
        operation="chat",
    )
    if handler:
        handler.start_llm(llm_invocation)

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
                    "agent_type": "specialist",
                    "session_id": state["session_id"],
                },
                "callbacks": [callback],
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

    # Update LLMInvocation with token counts and messages
    llm_invocation.input_tokens = callback.tokens.get("input", 0)
    llm_invocation.output_tokens = callback.tokens.get("output", 0)
    llm_invocation.input_messages = callback.input_messages
    llm_invocation.output_messages = callback.output_messages

    # Stop LLMInvocation
    if handler:
        handler.stop_llm(llm_invocation)

    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "hotel_specialist"

    # Stop AgentInvocation
    if handler:
        handler.stop_agent(agent_invocation)

    return state


def hotel_specialist_node(state: PlannerState) -> PlannerState:
    handler = state.get("telemetry_handler")

    # Create and start AgentInvocation
    agent_invocation = AgentInvocation(
        name="hotel_specialist",
        agent_type="specialist",
        framework="langgraph",
        model=_model_name(),
        tools=["mock_search_hotels"],
    )
    if handler:
        handler.start_agent(agent_invocation)

    # Create callback for token and tool tracking
    callback = TelemetryCallback(handler)

    # Create and start LLMInvocation
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        provider="openai",
        framework="langgraph",
        operation="chat",
    )
    if handler:
        handler.start_llm(llm_invocation)

    llm = _create_llm(
        "hotel_specialist", temperature=0.3, session_id=state["session_id"]
    )
    
    agent = (
        _create_react_agent(llm, tools=[mock_search_hotels]).with_config(
            {
                "run_name": "hotel_specialist",
                "tags": ["agent", "agent:hotel_specialist"],
                "metadata": {
                    "agent_name": "hotel_specialist",
                    "agent_type": "specialist",
                    "session_id": state["session_id"],
                },
                "callbacks": [callback],
            }
        )
    )
    step = (
        f"Recommend hotels in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    state["hotel_summary"] = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )

    # Update LLMInvocation with token counts and messages
    llm_invocation.input_tokens = callback.tokens.get("input", 0)
    llm_invocation.output_tokens = callback.tokens.get("output", 0)
    llm_invocation.input_messages = callback.input_messages
    llm_invocation.output_messages = callback.output_messages

    # Stop LLMInvocation
    if handler:
        handler.stop_llm(llm_invocation)

    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "activity_specialist"

    # Stop AgentInvocation
    if handler:
        handler.stop_agent(agent_invocation)

    return state



def activity_specialist_node(state: PlannerState) -> PlannerState:
    handler = state.get("telemetry_handler")

    # Create and start AgentInvocation
    agent_invocation = AgentInvocation(
        name="activity_specialist",
        agent_type="specialist",
        framework="langgraph",
        model=_model_name(),
        tools=["mock_search_activities"],
    )
    if handler:
        handler.start_agent(agent_invocation)

    # Create callback for token and tool tracking
    callback = TelemetryCallback(handler)

    # Create and start LLMInvocation
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        provider="openai",
        framework="langgraph",
        operation="chat",
    )
    if handler:
        handler.start_llm(llm_invocation)

    llm = _create_llm(
        "activity_specialist", temperature=0.5, session_id=state["session_id"]
    )
    
    agent = (
        _create_react_agent(llm, tools=[mock_search_activities]).with_config(
            {
                "run_name": "activity_specialist",
                "tags": ["agent", "agent:activity_specialist"],
                "metadata": {
                    "agent_name": "activity_specialist",
                    "agent_type": "specialist",
                    "session_id": state["session_id"],
                },
                "callbacks": [callback],
            }
        )
    )
    step = (
        f"Recommend activities in {state['destination']} for {state['travellers']} "
        f"travellers from {state['departure']} to {state['return_date']}."
    )
    result = agent.invoke({"messages": [HumanMessage(content=step)]})
    final_message = result["messages"][-1]
    state["activity_summary"] = (
        final_message.content
        if isinstance(final_message, BaseMessage)
        else str(final_message)
    )

    # Update LLMInvocation with token counts and messages
    llm_invocation.input_tokens = callback.tokens.get("input", 0)
    llm_invocation.output_tokens = callback.tokens.get("output", 0)
    llm_invocation.input_messages = callback.input_messages
    llm_invocation.output_messages = callback.output_messages

    # Stop LLMInvocation
    if handler:
        handler.stop_llm(llm_invocation)

    state["messages"].append(
        final_message
        if isinstance(final_message, BaseMessage)
        else AIMessage(content=str(final_message))
    )
    state["current_agent"] = "plan_synthesizer"

    # Stop AgentInvocation
    if handler:
        handler.stop_agent(agent_invocation)

    return state



def plan_synthesizer_node(state: PlannerState) -> PlannerState:
    handler = state.get("telemetry_handler")

    # Create and start AgentInvocation
    agent_invocation = AgentInvocation(
        name="plan_synthesizer",
        agent_type="synthesizer",
        framework="langgraph",
        model=_model_name(),
    )
    if handler:
        handler.start_agent(agent_invocation)

    # Create callback for token and tool tracking
    callback = TelemetryCallback(handler)

    # Create and start LLMInvocation
    llm_invocation = LLMInvocation(
        request_model=_model_name(),
        provider="openai",
        framework="langgraph",
        operation="chat",
    )
    if handler:
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
        ],
        config={"callbacks": [callback]},
    )
    state["final_itinerary"] = response.content
    state["messages"].append(response)
    state["current_agent"] = "completed"

    # Update LLMInvocation with token counts and messages
    llm_invocation.input_tokens = callback.tokens.get("input", 0)
    llm_invocation.output_tokens = callback.tokens.get("output", 0)
    llm_invocation.input_messages = callback.input_messages
    llm_invocation.output_messages = callback.output_messages

    # Stop LLMInvocation
    if handler:
        handler.stop_llm(llm_invocation)

    # Stop AgentInvocation
    if handler:
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
    
    _configure_otlp_tracing()
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
            result = run_travel_planner(user_request)
            print(f"✅ Planning completed for session: {result['session_id']}")

            wait_seconds = int(os.getenv("EVAL_WAIT_SECONDS", "150"))
            if wait_seconds > 0:
                print(f"\n⏳ Waiting {wait_seconds} seconds for evaluations to finish...")
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
