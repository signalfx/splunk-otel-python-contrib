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
Multi-agent travel planner demonstrating OpenAI Agents v2 instrumentation.

Uses the native OpenAI Agents SDK with multiple specialized agents to build
a travel itinerary, demonstrating OpenTelemetry instrumentation with GenAI
semantic conventions.

Agents:
- Flight Specialist: Searches for flights
- Hotel Specialist: Recommends accommodations
- Activity Specialist: Curates activities
- Travel Coordinator: Orchestrates and synthesizes the plan

See README.md for more information
"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from typing import Any  # noqa: E402

# Load environment variables FIRST before any other imports
# This ensures OTEL_SERVICE_NAME and other env vars are available when SDK initializes
from dotenv import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from typing import Any  # noqa: E402

from agents import (  # noqa: E402
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    trace,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from opentelemetry import _events, _logs, metrics, trace as otel_trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (  # noqa: E402
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # noqa: E402
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: E402
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.openai_agents import (  # noqa: E402
    OpenAIAgentsInstrumentor,
)
from opentelemetry.sdk._events import EventLoggerProvider  # noqa: E402
from opentelemetry.sdk._logs import LoggerProvider  # noqa: E402
from opentelemetry.sdk._logs.export import (  # noqa: E402
    BatchLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import (  # noqa: E402
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402

from util import OAuth2TokenManager  # noqa: E402

# ---------------------------------------------------------------------------
# LLM Configuration - OAuth2 Provider
# ---------------------------------------------------------------------------

# Optional app key for request tracking
LLM_APP_KEY = os.environ.get("LLM_APP_KEY")

# Model name from environment
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")

# Check if we should use OAuth2 or standard OpenAI
USE_OAUTH2 = bool(os.environ.get("LLM_CLIENT_ID"))

# Initialize token manager if OAuth2 credentials are present
token_manager: OAuth2TokenManager | None = None
if USE_OAUTH2:
    token_manager = OAuth2TokenManager()
    print("[AUTH] Using OAuth2 authentication")
else:
    print("[AUTH] Using standard OpenAI API key")


def get_openai_client() -> AsyncOpenAI:
    """Create OpenAI client with fresh OAuth2 token or standard API key."""
    if USE_OAUTH2 and token_manager:
        token = token_manager.get_token()
        base_url = OAuth2TokenManager.get_llm_base_url(OPENAI_MODEL)

        # Build extra headers
        extra_headers: dict[str, str] = {"api-key": token}
        if LLM_APP_KEY:
            extra_headers["x-app-key"] = LLM_APP_KEY

        return AsyncOpenAI(
            api_key="placeholder",
            base_url=base_url,
            default_headers=extra_headers,
        )
    else:
        # Standard OpenAI client using OPENAI_API_KEY
        return AsyncOpenAI()


class CustomChatCompletionsModel(OpenAIChatCompletionsModel):
    """Custom ChatCompletions model that adds 'user' field with app key for OAuth2 endpoints."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        app_key: str | None = None,
    ):
        super().__init__(model=model, openai_client=openai_client)
        # Some LLM APIs require user field as JSON: {"appkey": "<value>"}

        self._user = json.dumps({"appkey": app_key or ""})

    async def _fetch_response(self, *args, **kwargs):
        # Get the original client
        client = self._get_client()

        # Create a wrapped chat completions that adds user field
        original_create = client.chat.completions.create

        async def create_with_user(*create_args, **create_kwargs):
            create_kwargs["user"] = self._user
            return await original_create(*create_args, **create_kwargs)

        # Temporarily replace the create method
        client.chat.completions.create = create_with_user
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            # Restore original method
            client.chat.completions.create = original_create


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

DESTINATIONS = {
    "paris": {
        "highlights": [
            "Eiffel Tower at sunset",
            "Seine dinner cruise",
            "Day trip to Versailles",
        ],
    },
    "tokyo": {
        "highlights": [
            "Tsukiji market food tour",
            "Ghibli Museum visit",
            "Day trip to Hakone hot springs",
        ],
    },
    "rome": {
        "highlights": [
            "Colosseum underground tour",
            "Private pasta masterclass",
            "Sunset walk through Trastevere",
        ],
    },
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


@function_tool
def search_flights(origin: str, destination: str, departure_date: str) -> str:
    """Search for flight options between origin and destination."""
    random.seed(hash((origin, destination, departure_date)) % (2**32))
    airline = random.choice(["SkyLine", "AeroJet", "CloudNine"])
    fare = random.randint(700, 1250)
    return (
        f"Top choice: {airline} non-stop service {origin}->{destination}, "
        f"depart {departure_date} 09:15, arrive same day 17:05. "
        f"Premium economy fare ${fare} return."
    )


@function_tool
def search_hotels(destination: str, check_in: str, check_out: str) -> str:
    """Search for hotel recommendations at the destination."""
    random.seed(hash((destination, check_in, check_out)) % (2**32))
    name = random.choice(["Grand Meridian", "Hotel Lumière", "The Atlas"])
    rate = random.randint(240, 410)
    return (
        f"{name} near the historic centre. Boutique suites, rooftop bar, "
        f"average nightly rate ${rate} including breakfast."
    )


@function_tool
def search_activities(destination: str) -> str:
    """Get signature activities and experiences for a destination."""
    data = DESTINATIONS.get(destination.lower(), DESTINATIONS["paris"])
    bullets = "\n".join(f"- {item}" for item in data["highlights"])
    return f"Signature experiences in {destination.title()}:\n{bullets}"


# ---------------------------------------------------------------------------
# OpenTelemetry configuration
# ---------------------------------------------------------------------------


def configure_otel() -> None:
    """Configure OpenTelemetry SDK for traces, metrics, and logs."""
    # Create resource with service name from environment (OTEL_SERVICE_NAME)
    # Resource.create() automatically picks up OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES
    resource = Resource.create()

    # Traces
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    otel_trace.set_tracer_provider(trace_provider)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

    # Logs
    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )

    # Events
    _events.set_event_logger_provider(EventLoggerProvider())

    # OpenAI Agents instrumentation
    OpenAIAgentsInstrumentor().instrument(tracer_provider=trace_provider)

    # Set default OpenAI client for agents
    client = get_openai_client()
    set_default_openai_client(client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def get_chat_completions_model() -> OpenAIChatCompletionsModel:
    """Create a ChatCompletions model using the configured OpenAI client."""
    client = get_openai_client()
    if USE_OAUTH2:
        # Use custom model that adds 'user' field with appkey for OAuth2 endpoints
        return CustomChatCompletionsModel(
            model=OPENAI_MODEL, openai_client=client, app_key=LLM_APP_KEY
        )
    return OpenAIChatCompletionsModel(model=OPENAI_MODEL, openai_client=client)


def create_flight_agent() -> Agent:
    """Create the flight specialist agent."""
    return Agent(
        name="Flight Specialist",
        model=get_chat_completions_model(),
        instructions=(
            "You are a flight specialist. Search for the best flight options "
            "using the search_flights tool. Provide clear recommendations including "
            "airline, schedule, and fare information."
        ),
        tools=[search_flights],
    )


def create_hotel_agent() -> Agent:
    """Create the hotel specialist agent."""
    return Agent(
        name="Hotel Specialist",
        model=get_chat_completions_model(),
        instructions=(
            "You are a hotel specialist. Find the best accommodation using the "
            "search_hotels tool. Provide detailed recommendations including "
            "location, amenities, and pricing."
        ),
        tools=[search_hotels],
    )


def create_activity_agent() -> Agent:
    """Create the activity specialist agent."""
    return Agent(
        name="Activity Specialist",
        model=get_chat_completions_model(),
        instructions=(
            "You are an activities specialist. Curate memorable experiences using "
            "the search_activities tool. Provide detailed activity recommendations "
            "that match the traveler's interests."
        ),
        tools=[search_activities],
    )


def create_coordinator_agent() -> Agent:
    """Create the travel coordinator agent that synthesizes the final itinerary."""
    return Agent(
        name="Travel Coordinator",
        model=get_chat_completions_model(),
        instructions=(
            "You are a travel coordinator. Synthesize flight, hotel, and activity information "
            "into a comprehensive, well-organized travel itinerary with clear sections."
        ),
    )


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_travel_planner() -> None:
    """Execute the multi-agent travel planning workflow."""
    # Sample travel request
    origin = "Seattle"
    destination = "Paris"
    departure = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    return_date = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")

    print("🌍 Multi-Agent Travel Planner")
    print("=" * 60)
    print(f"\nOrigin: {origin}")
    print(f"Destination: {destination}")
    print(f"Dates: {departure} to {return_date}\n")
    print("=" * 60)

    # Create all specialist agents and coordinator
    flight_agent = create_flight_agent()
    hotel_agent = create_hotel_agent()
    activity_agent = create_activity_agent()
    coordinator = create_coordinator_agent()

    initial_request = f"Plan a romantic week-long trip from {origin} to {destination}, departing {departure} and returning {return_date}"
    print(f"\nRequest: {initial_request}\n")
    metadata: dict[str, Any] = {
        "initial_request": initial_request,
    }

    final_output = None
    try:
        with trace("Travel planner workflow", metadata=metadata):
            # Step 1: Flight Specialist
            print("\n✈️  Flight Specialist - Searching for flights...")
            flight_result = Runner.run_sync(
                flight_agent,
                f"Find flights from {origin} to {destination} departing {departure}",
            )
            flight_info = flight_result.final_output
            print(f"Result: {flight_info[:200]}...\n")

            # Step 2: Hotel Specialist
            print("🏨 Hotel Specialist - Searching for hotels...")
            hotel_result = Runner.run_sync(
                hotel_agent,
                f"Find a boutique hotel in {destination}, check-in {departure}, check-out {return_date}",
            )
            hotel_info = hotel_result.final_output
            print(f"Result: {hotel_info[:200]}...\n")

            # Step 3: Activity Specialist
            print("🎭 Activity Specialist - Curating activities...")
            activity_result = Runner.run_sync(
                activity_agent,
                f"Find unique activities and experiences in {destination}",
            )
            activity_info = activity_result.final_output
            print(f"Result: {activity_info[:200]}...\n")

            # Step 4: Coordinator - Synthesize final itinerary
            print("📝 Coordinator - Creating final itinerary...")
            synthesis_prompt = f"""
Create a comprehensive travel itinerary with the following information:

FLIGHTS:
{flight_info}

ACCOMMODATION:
{hotel_info}

ACTIVITIES:
{activity_info}

Please organize this into a clear, well-formatted itinerary for a romantic week-long trip.
"""

            final_result = Runner.run_sync(coordinator, synthesis_prompt)
            final_output = final_result.final_output

            print("\n" + "=" * 60)
            print("✅ Travel Itinerary Complete!")
            print("=" * 60)
            print(f"\n{final_output}\n")

    finally:
        # Flush telemetry
        flush_telemetry()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def flush_telemetry() -> None:
    """Force flush all telemetry providers before exit."""
    print("\n" + "=" * 80)
    print("TELEMETRY OUTPUT BELOW")
    print("=" * 80)
    print("\n[FLUSH] Starting telemetry flush")

    # Flush traces
    tracer_provider = otel_trace.get_tracer_provider()
    if hasattr(tracer_provider, "force_flush"):
        print("[FLUSH] Flushing traces (timeout=30s)")
        tracer_provider.force_flush(timeout_millis=30000)

    # Flush metrics
    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, "force_flush"):
        print("[FLUSH] Flushing metrics (timeout=30s)")
        meter_provider.force_flush(timeout_millis=30000)

    # Flush logs
    logger_provider = _logs.get_logger_provider()
    if hasattr(logger_provider, "force_flush"):
        print("[FLUSH] Flushing logs (timeout=30s)")
        logger_provider.force_flush(timeout_millis=30000)

    # Small delay for network buffers
    time.sleep(2)
    print("[FLUSH] Telemetry flush complete")


def main(manual_instrumentation: bool = False) -> None:
    """Main entry point for the travel planner example."""
    # Note: load_dotenv() is called at module level before imports

    if manual_instrumentation:
        configure_otel()
        print("✓ Manual OpenTelemetry instrumentation configured")

    run_travel_planner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent travel planner example"
    )
    parser.add_argument(
        "--manual-instrumentation",
        action="store_true",
        help="Use manual instrumentation (for debugging)",
    )
    args = parser.parse_args()

    main(manual_instrumentation=args.manual_instrumentation)
