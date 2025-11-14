"""LiteLLM-backed multi-agent travel booking demo.

This sample clones the Traceloop/LangGraph travel supervisor demo but routes all
Chat Completions through a LiteLLM proxy (OpenAI-compatible). Point
``LITELLM_BASE_URL`` at your local or cluster proxy endpoint and the demo will
drive LiteLLM instead of talking to CircuIT directly.
"""

import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from mcp_use import MCPClient

load_dotenv()

# ---------------------------------------------------------------------------
# LiteLLM configuration (env driven)
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
DEFAULT_PROMPT = os.getenv(
    "TRAVEL_DEMO_PROMPT",
    "book a flight from BOS to JFK and a stay at McKittrick Hotel",
)

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL)
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_TIMEOUT = int(os.getenv("LITELLM_TIMEOUT", "60"))
LITELLM_MAX_RETRIES = int(os.getenv("LITELLM_MAX_RETRIES", "2"))
LITELLM_TEMPERATURE = float(os.getenv("LITELLM_TEMPERATURE", "0.1"))

if not LITELLM_API_KEY:
    raise RuntimeError(
        "LITELLM_API_KEY not set. Export it (see .env.example) so ChatOpenAI can "
        "authenticate against the LiteLLM proxy."
    )


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")
        # Be defensive: some node updates can be None or omit the 'messages' key.
        if not node_update:
            print("(no update payload from node)")
            print("\n")
            continue

        raw_messages = node_update.get("messages") if isinstance(node_update, dict) else None
        if raw_messages is None:
            print("(node update contains no 'messages')")
            print("\n")
            continue

        messages = convert_to_messages(raw_messages)
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

async def initialize_mcp_client():
    """Initialize the MCP client"""
    config = {
        "mcpServers": {
            "time": {
                "command": "uvx",
                "args": ["mcp-server-time"]
            }
        }
    }
    client = MCPClient.from_dict(config)
    await client.start_all_sessions()
    return client

@tool
async def get_current_time_mcp(timezone: str = "America/New_York") -> str:
    """Get current time in specified timezone or the system timezone. 
    
    Args:
        timezone: IANA timezone name (str, optional). Examples: 'America/New_York', 'Europe/London', 'Asia/Tokyo'
    """
    Traceloop.set_association_properties({
        "mcp.tool": "get_current_time",
        "mcp.timezone": timezone,
        "agent.type": "time_agent"
    })
    mcp_client = await initialize_mcp_client()
    params = {} if timezone is None else {"timezone": timezone}
    result = await mcp_client.call("time", "get_current_time", params)
    Traceloop.set_association_properties({
            "mcp.response.timezone": result.get('timezone'),
            "mcp.response.datetime": result.get('datetime'),
            "mcp.response.is_dst": str(result.get('is_dst'))
        })
    return f"Current time in {result['timezone']}: {result['datetime']} (DST: {result['is_dst']})"

@tool
def book_hotel(hotel_name: str):
    """Book a hotel"""
    Traceloop.set_association_properties({
        "booking.type": "hotel",
        "booking.hotel_name": hotel_name,
        "agent.name": "hotel_assistant"
    })
    return f"Successfully booked a stay at {hotel_name}."

@tool
def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    Traceloop.set_association_properties({
        "booking.type": "flight",
        "booking.from_airport": from_airport,
        "booking.to_airport": to_airport,
        "agent.name": "flight_assistant"
    })
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

def _build_chat_model() -> ChatOpenAI:
    """Instantiate ChatOpenAI pointing at LiteLLM."""
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=LITELLM_TEMPERATURE,
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_BASE_URL,
        timeout=LITELLM_TIMEOUT,
        max_retries=LITELLM_MAX_RETRIES,
    )


def build_supervisor() -> Any:
    """Compile the LangGraph supervisor bound to the LiteLLM chat model."""
    model = _build_chat_model()
    # NOTE: tools are temporarily disabled here to avoid sending
    # 'parallel_tool_calls' into the upstream LLM client used by LiteLLM
    # which causes an unexpected keyword argument error in some
    # OpenAI client versions. Restore tools=[book_flight, get_current_time_mcp]
    # once LiteLLM/openai client compatibility is resolved.
    flight_assistant = create_react_agent(
        model=model,
        tools=[],
        prompt=(
            "You are a flight booking assistant. When getting the current time, you MUST "
            "specify a timezone parameter. Use 'America/New_York' for US East Coast."
        ),
        name="flight_assistant",
    )

    # NOTE: tools disabled for hotel assistant for the same compatibility reason.
    hotel_assistant = create_react_agent(
        model=model,
        tools=[],
        prompt="You are a hotel booking assistant.",
        name="hotel_assistant",
    )

    return create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=model,
        prompt=(
            "You manage a hotel booking assistant and a flight booking assistant. Assign work"
            " to them based on the user request."
        ),
    ).compile()


def run_demo(supervisor: Any, user_prompt: str) -> None:
    """Stream the supervisor output for the provided prompt."""
    payload = {"messages": [{"role": "user", "content": user_prompt}]}
    for chunk in supervisor.stream(payload):
        pretty_print_messages(chunk)
        print("\n")


def main() -> None:
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    demo_supervisor = build_supervisor()
    run_demo(demo_supervisor, user_prompt)


if __name__ == "__main__":
    main()
