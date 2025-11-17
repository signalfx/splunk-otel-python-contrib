#!/usr/bin/env python3
"""Minimal LangChain client that pings LiteLLM for a sanity check."""

import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
DEFAULT_PROMPT = os.getenv(
    "LITELLM_SMOKE_PROMPT",
    "Say hello and confirm you reached me via LiteLLM.",
)

BASE_URL = os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL)
API_KEY = os.getenv("LITELLM_API_KEY")
TIMEOUT = int(os.getenv("LITELLM_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("LITELLM_MAX_RETRIES", "2"))
TEMPERATURE = float(os.getenv("LITELLM_TEMPERATURE", "0.2"))

if not API_KEY:
    raise SystemExit(
        "LITELLM_API_KEY is not set. Export it (see .env.example) before running the smoke test."
    )


def build_chat_model() -> ChatOpenAI:
    """Create a ChatOpenAI client pointed at LiteLLM."""
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE,
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def run(prompt: str) -> None:
    """Send the prompt through LiteLLM and print the assistant's reply."""
    chat = build_chat_model()
    messages: List[HumanMessage] = [HumanMessage(content=prompt)]
    print(f"Sending prompt to LiteLLM @ {BASE_URL} using model '{DEFAULT_MODEL}'...\n")
    response = chat.invoke(messages)
    print("--- LiteLLM response ---")
    print(response.content)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    run(prompt)
