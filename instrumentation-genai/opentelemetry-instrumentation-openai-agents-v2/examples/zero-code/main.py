"""Zero-code OpenAI Agents example with OAuth2 token support."""

from __future__ import annotations

import os

from agents import Agent, Runner, function_tool, set_default_openai_client
from dotenv import load_dotenv
from openai import OpenAI

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from util import OAuth2TokenManager

# =============================================================================
# LLM Configuration - OAuth2 Provider
# =============================================================================

# Optional app key for request tracking
LLM_APP_KEY = os.environ.get("LLM_APP_KEY")

# Check if we should use OAuth2 or standard OpenAI
USE_OAUTH2 = bool(os.environ.get("LLM_CLIENT_ID"))

# Initialize token manager if OAuth2 credentials are present
token_manager: OAuth2TokenManager | None = None
if USE_OAUTH2:
    token_manager = OAuth2TokenManager()


def get_openai_client() -> OpenAI:
    """Create OpenAI client with fresh OAuth2 token or standard API key."""
    if USE_OAUTH2 and token_manager:
        token = token_manager.get_token()
        base_url = OAuth2TokenManager.get_llm_base_url(
            os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        )

        # Build extra headers
        extra_headers: dict[str, str] = {"api-key": token}
        if LLM_APP_KEY:
            extra_headers["x-app-key"] = LLM_APP_KEY

        return OpenAI(
            api_key="placeholder",  # Required but we use api-key header
            base_url=base_url,
            default_headers=extra_headers,
        )
    else:
        # Standard OpenAI API
        return OpenAI()


def configure_tracing() -> None:
    """Ensure tracing exports spans even without auto-instrumentation."""

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)

    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather response for the requested city."""

    return f"The forecast for {city} is sunny with pleasant temperatures."


def run_agent() -> None:
    """Create a simple agent and execute a single run."""

    # Configure OpenAI client with OAuth2 or standard API key
    client = get_openai_client()
    set_default_openai_client(client)

    if USE_OAUTH2:
        print("[AUTH] Using OAuth2 authentication")
    else:
        print("[AUTH] Using standard OpenAI API key")

    assistant = Agent(
        name="Travel Concierge",
        instructions=(
            "You are a concise travel concierge. Use the weather tool when the"
            " traveler asks about local conditions."
        ),
        tools=[get_weather],
    )

    result = Runner.run_sync(
        assistant,
        "I'm visiting Barcelona this weekend. How should I pack?",
    )

    print("\n[SUCCESS] Agent execution completed")
    print("Agent response:")
    print(result.final_output)


def main() -> None:
    load_dotenv()
    configure_tracing()
    run_agent()


if __name__ == "__main__":
    main()
