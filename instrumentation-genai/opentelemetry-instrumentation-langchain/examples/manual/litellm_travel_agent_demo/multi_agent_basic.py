#!/usr/bin/env python3
"""Simplified multi-agent travel planner powered by LiteLLM."""

import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

load_dotenv()

class LiteLLMChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that strips unsupported args before hitting LiteLLM."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        kwargs.pop("parallel_tool_calls", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        kwargs.pop("parallel_tool_calls", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
DEFAULT_PROMPT = os.getenv(
    "TRAVEL_DEMO_PROMPT",
    "book a flight from BOS to JFK and reserve a hotel near downtown Manhattan",
)

BASE_URL = os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL)
API_KEY = os.getenv("LITELLM_API_KEY")
TIMEOUT = int(os.getenv("LITELLM_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("LITELLM_MAX_RETRIES", "2"))
TEMPERATURE = float(os.getenv("LITELLM_TEMPERATURE", "0.1"))

if not API_KEY:
    raise SystemExit(
        "LITELLM_API_KEY is not set. Export it (see .env.example) before running the basic planner."
    )


def build_chat_model() -> ChatOpenAI:
    """Instantiate ChatOpenAI pointing at LiteLLM."""
    return LiteLLMChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE,
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def build_graph() -> Any:
    """Compile the LangGraph supervisor bound to LiteLLM."""
    model = build_chat_model()

    # NOTE: Tools are disabled because LiteLLM's upstream currently rejects the
    # OpenAI `parallel_tool_calls` parameter that LangChain sends when tools are
    # registered. This mirrors the workaround in main.py.
    flight_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a flight specialist. Ask for missing airport info and respond with"
            " clear flight options using structured bullet points."
        ),
        name="flight_assistant",
    )

    hotel_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a hotel specialist focused on boutique stays. Provide two options"
            " max and mention notable amenities."
        ),
        name="hotel_assistant",
    )

    supervisor = create_supervisor(
        agents=[flight_agent, hotel_agent],
        model=model,
        prompt=(
            "You are a friendly travel coordinator. Understand the request, call the"
            " flight_assistant for all flight needs and the hotel_assistant for lodging."
            " Respond with a concise itinerary once all bookings are complete."
        ),
    )
    return supervisor.compile()


def stream_run(prompt: str) -> None:
    """Run the planner and stream intermediate updates to stdout."""
    graph = build_graph()
    print(f"User request: {prompt}\n")

    final_message = None
    for update in graph.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"recursion_limit": 20},
    ):
        if "__end__" in update:
            messages = convert_to_messages(update["__end__"].get("messages", []))
            if messages:
                final_message = messages[-1]
            continue

        for node, payload in update.items():
            print(f"[{node}] update")
            if not payload:
                print("(no payload)")
                print()
                continue
            messages = convert_to_messages(payload.get("messages", []))
            if messages:
                print(messages[-1].pretty_repr())
            print()

    if final_message:
        print("Final itinerary:\n")
        print(final_message.pretty_repr())


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    stream_run(task)
