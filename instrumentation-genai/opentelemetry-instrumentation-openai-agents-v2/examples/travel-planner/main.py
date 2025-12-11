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

import argparse
import random
import time
from datetime import datetime, timedelta

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.openai_agents.span_processor import (
    start_multi_agent_workflow,
    stop_multi_agent_workflow,
)
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

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
    # Traces
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(trace_provider)

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


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def create_flight_agent() -> Agent:
    """Create the flight specialist agent."""
    return Agent(
        name="Flight Specialist",
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

    # Start a global workflow that spans all agent calls
    initial_request = f"Plan a romantic week-long trip from {origin} to {destination}, departing {departure} and returning {return_date}"
    start_multi_agent_workflow(
        workflow_name="travel-planner",
        initial_input=initial_request,
        workflow_origin=origin,
        workflow_destination=destination,
    )

    final_output = None
    try:
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
        # Stop the global workflow with final output
        stop_multi_agent_workflow(final_output=final_output)

        # Allow time for telemetry to flush
        time.sleep(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(manual_instrumentation: bool = False) -> None:
    """Main entry point for the travel planner example."""
    load_dotenv()

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
