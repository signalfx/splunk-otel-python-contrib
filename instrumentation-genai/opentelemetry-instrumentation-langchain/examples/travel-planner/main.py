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
Simple LangChain travel-planner app for APMDT-3098.

Produces a steady trickle of gen_ai.* spans (LLM invocation + tool call +
retrieval) to keep the Spark stream alive.  Runs a short round of travel
Q&A on each invocation — no evals, no heavy dependencies.

Span coverage per run:
  - retrieval  (in-memory travel knowledge base)
  - chat       (specialist LLM — may call tools)
  - execute_tool (mock_search_flights)
  - execute_tool (mock_search_activities)
  - chat       (synthesizer LLM)
"""

from __future__ import annotations

import base64
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Annotated, Any, List, Optional, TypedDict
from uuid import uuid4

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import RetrievalInvocation

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("travel_planner")

# ---------------------------------------------------------------------------
# Circuit OAuth helper
# ---------------------------------------------------------------------------


def _get_circuit_token() -> str:
    client_id = os.environ["LLM_CLIENT_ID"]
    client_secret = os.environ["LLM_CLIENT_SECRET"]
    token_url = os.environ.get(
        "LLM_TOKEN_URL", "https://id.cisco.com/oauth2/default/v1/token"
    )
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        token_url,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _create_llm(
    agent_name: str, session_id: str, temperature: float = 0.3
) -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    base_url = os.getenv("LLM_BASE_URL", "https://chat-ai.cisco.com/openai/deployments")
    app_key = os.getenv("LLM_APP_KEY", "")
    token = _get_circuit_token()
    # Circuit requires the model name in the base URL path
    full_base_url = f"{base_url.rstrip('/')}/{model}"
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=token,
        base_url=full_base_url,
        default_headers={"api-key": token},
        model_kwargs={"user": f'{{"appkey":"{app_key}"}}'},
        tags=[f"agent:{agent_name}", "travel-planner"],
        metadata={
            "agent_name": agent_name,
            "session_id": session_id,
            "ls_model_name": model,
        },
    )


# ---------------------------------------------------------------------------
# Mock tools  (produce execute_tool spans via LangChain instrumentation)
# ---------------------------------------------------------------------------

DESTINATIONS = {
    "paris": [
        "Eiffel Tower at sunset",
        "Seine dinner cruise",
        "Day trip to Versailles",
    ],
    "tokyo": ["Tsukiji market food tour", "Ghibli Museum visit", "Day trip to Hakone"],
    "rome": [
        "Colosseum underground tour",
        "Private pasta masterclass",
        "Trastevere walk",
    ],
    "sydney": [
        "Harbour Bridge climb",
        "Bondi to Coogee coastal walk",
        "Opera House tour",
    ],
}


@tool
def mock_search_flights(origin: str, destination: str, departure: str) -> str:
    """Return mock flight options for a given route and date."""
    random.seed(hash((origin, destination, departure)) % (2**32))
    airline = random.choice(["SkyLine", "AeroJet", "CloudNine", "PacificAir"])
    fare = random.randint(650, 1400)
    return (
        f"{airline} non-stop {origin}→{destination}, depart {departure} 08:45, "
        f"arrive same day 18:30. Return premium economy ${fare}."
    )


@tool
def mock_search_activities(destination: str) -> str:
    """Return signature activities for the destination."""
    highlights = DESTINATIONS.get(destination.lower(), DESTINATIONS["paris"])
    return "Highlights:\n" + "\n".join(f"- {h}" for h in highlights)


# ---------------------------------------------------------------------------
# In-memory travel knowledge base  (produces retrieval spans)
# ---------------------------------------------------------------------------

TRAVEL_KB: dict[str, list[str]] = {
    "paris": [
        "Best time to visit Paris is April–June or September–October.",
        "A Navigo Week Pass covers all metro/bus zones for €30.",
        "Louvre is closed on Tuesdays; book timed-entry tickets in advance.",
        "Seine dinner cruises depart from Pont de l'Alma, cost ~€90/person.",
    ],
    "tokyo": [
        "Cherry blossom (sakura) season runs late March to mid April.",
        "IC cards (Suica/Pasmo) work on all trains, buses, and convenience stores.",
        "Ghibli Museum requires advance reservations via the Lawson ticketing system.",
        "Hakone day trips take ~85 min on the Romancecar from Shinjuku.",
    ],
    "rome": [
        "Book Colosseum skip-the-line tickets at least 2 weeks ahead in summer.",
        "Vatican Museums are closed on Sundays except the last Sunday of the month (free).",
        "Trastevere is the best neighbourhood for authentic Roman dining.",
        "Trenitalia connects Rome Termini to Naples in 70 min by high-speed rail.",
    ],
    "sydney": [
        "BridgeClimb bookings must be made at least 24 hours in advance.",
        "Bondi to Coogee coastal walk is 6 km, takes ~2 hours, no cost.",
        "Opal card is required for public transport; available at airport convenience stores.",
        "Blue Mountains day trip from Central Station takes ~2 hours.",
    ],
}


def retrieve_destination_context(destination: str, query: str) -> list[str]:
    """Retrieve relevant travel facts for a destination. Produces a retrieval span."""
    handler = get_telemetry_handler()
    retrieval = RetrievalInvocation(
        query=query,
        retriever_type="in_memory",
        request_model=None,
    )
    if handler:
        handler.start_retrieval(retrieval)

    docs = TRAVEL_KB.get(destination.lower(), TRAVEL_KB["paris"])

    if handler:
        handler.stop_retrieval(retrieval)
    return docs


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class TripState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    session_id: str
    origin: str
    destination: str
    departure: str
    return_date: str
    retrieved_context: Optional[str]
    flight_summary: Optional[str]
    activities_summary: Optional[str]
    final_plan: Optional[str]


def _compute_dates() -> tuple[str, str]:
    start = datetime.now() + timedelta(days=30)
    return start.strftime("%Y-%m-%d"), (start + timedelta(days=7)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def retrieval_node(state: TripState) -> TripState:
    """Retrieve destination context from the in-memory KB (produces a retrieval span)."""
    query = f"travel tips for {state['destination']}"
    docs = retrieve_destination_context(state["destination"], query)
    state["retrieved_context"] = "\n".join(f"- {d}" for d in docs)
    return state


def specialist_node(state: TripState) -> TripState:
    """Single specialist: fetches flight + activity data via tools, then summarises."""
    sid = state["session_id"]
    llm = _create_llm("specialist", sid, temperature=0.4)
    llm_with_tools = llm.bind_tools([mock_search_flights, mock_search_activities])

    context = state.get("retrieved_context") or ""
    prompt = (
        f"Plan a 7-day trip from {state['origin']} to {state['destination']} "
        f"departing {state['departure']}. Use the available tools to find a flight "
        f"and activities, then summarise your findings briefly."
    )
    messages = [
        SystemMessage(
            content=(
                "You are a travel specialist. Use tools to gather data, then summarise.\n\n"
                f"Destination knowledge base:\n{context}"
            )
        ),
        HumanMessage(content=prompt),
    ]

    # First LLM call — may return tool calls
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Execute any tool calls
    for tc in getattr(response, "tool_calls", []):
        selected = {
            "mock_search_flights": mock_search_flights,
            "mock_search_activities": mock_search_activities,
        }.get(tc["name"])
        if selected:
            tool_result = selected.invoke(tc)
            messages.append(tool_result)

    # Second LLM call — synthesise tool results
    if getattr(response, "tool_calls", []):
        final = llm_with_tools.invoke(messages)
        messages.append(final)
        summary = (
            final.content if isinstance(final.content, str) else str(final.content)
        )
    else:
        summary = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    state["flight_summary"] = summary
    state["activities_summary"] = summary
    state["messages"].extend([m for m in messages if isinstance(m, BaseMessage)])
    return state


def synthesizer_node(state: TripState) -> TripState:
    """Final LLM call: produce a concise itinerary from the specialist summary."""
    sid = state["session_id"]
    llm = _create_llm("synthesizer", sid, temperature=0.3)

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a travel planner. Produce a concise 3-day itinerary."
            ),
            HumanMessage(
                content=(
                    f"Trip: {state['origin']} → {state['destination']}, "
                    f"{state['departure']} to {state['return_date']}.\n\n"
                    f"Specialist notes:\n{state.get('flight_summary', '')}"
                )
            ),
        ]
    )
    state["final_plan"] = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    state["messages"].append(response)
    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    g = StateGraph(TripState)
    g.add_node("retrieval", retrieval_node)
    g.add_node("specialist", specialist_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_edge(START, "retrieval")
    g.add_edge("retrieval", "specialist")
    g.add_edge("specialist", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Scenarios — rotate through destinations to vary telemetry
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("Seattle", "Paris"),
    ("New York", "Tokyo"),
    ("San Francisco", "Rome"),
    ("Chicago", "Sydney"),
]


def run_once() -> None:
    session_id = str(uuid4())
    origin, destination = random.choice(SCENARIOS)
    departure, return_date = _compute_dates()

    logger.info(
        "travel-planner-ao run started session=%s route=%s->%s departure=%s",
        session_id,
        origin,
        destination,
        departure,
    )

    app = build_graph()
    initial: TripState = {
        "messages": [
            HumanMessage(content=f"Plan a trip from {origin} to {destination}.")
        ],
        "session_id": session_id,
        "origin": origin,
        "destination": destination,
        "departure": departure,
        "return_date": return_date,
        "retrieved_context": None,
        "flight_summary": None,
        "activities_summary": None,
        "final_plan": None,
    }

    final = app.invoke(initial, config={"configurable": {"thread_id": session_id}})
    plan = final.get("final_plan", "")
    logger.info(
        "travel-planner-ao run completed session=%s plan_chars=%d",
        session_id,
        len(plan),
    )
    if plan:
        logger.info("plan_preview=%s", plan[:300] + ("..." if len(plan) > 300 else ""))


if __name__ == "__main__":
    run_once()
