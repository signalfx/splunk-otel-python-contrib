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
Travel Assistant using OpenAI SDK with Tool Calling and Embeddings

This example demonstrates:
1. OpenAI SDK chat completions with tool calling
2. OpenAI SDK embeddings for semantic destination search
3. Multi-turn conversation with tool execution loop
4. Full observability with OpenTelemetry instrumentation
5. OAuth2 authentication for internal LLM gateway (Circuit)

Authentication is auto-detected:
- If LLM_CLIENT_ID is set: Uses OAuth2 (Circuit internal LLM gateway)
- Otherwise: Uses standard OpenAI API with OPENAI_API_KEY

Usage:
    # With standard OpenAI API:
    export OPENAI_API_KEY="your-api-key"

    # Or with Circuit (internal LLM gateway):
    export LLM_CLIENT_ID="your-client-id"
    export LLM_CLIENT_SECRET="your-client-secret"
    export LLM_APP_KEY="your-app-key"  # optional

    python main.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv

# Import OpenAI clients - instrumentation happens in setup_telemetry()
from openai import AzureOpenAI, OpenAI

from opentelemetry.trace import SpanKind

load_dotenv()

# Module-level globals that get initialized in setup_telemetry()
trace_provider = None
tracer = None
logger_provider = None
OTLP_ENDPOINT = None


def setup_telemetry():
    """Setup OpenTelemetry instrumentation.

    This function MUST be called from within if __name__ == "__main__":
    to prevent multiprocessing spawn from re-running the setup.
    """
    global trace_provider, tracer, logger_provider, OTLP_ENDPOINT

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
    from opentelemetry.sdk._events import EventLoggerProvider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    OTLP_ENDPOINT = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    resource = Resource.create(
        {
            "service.name": "travel_assistant_openai_shuwpan",
            "service.version": "1.0.0",
        }
    )

    # Configure Tracing
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
        )
    )
    trace.set_tracer_provider(trace_provider)
    tracer = trace.get_tracer("travel_assistant")

    # Configure Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    )
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[metric_reader])
    )

    # Configure Logging and Events
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=True)
        )
    )
    _logs.set_logger_provider(logger_provider)
    _events.set_event_logger_provider(
        EventLoggerProvider(logger_provider=logger_provider)
    )

    print(
        f"📡 Exporting Traces, Metrics, Logs to Console + OTLP ({OTLP_ENDPOINT})"
    )

    # Instrument OpenAI
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument(
        tracer_provider=trace_provider,
        meter_provider=metrics.get_meter_provider(),
        logger_provider=logger_provider,
    )
    print("✅ OpenAI instrumentation enabled")

    return trace_provider, tracer, logger_provider, OTLP_ENDPOINT


def create_openai_client() -> OpenAI:
    """Create OpenAI client for chat completions.

    Auto-detects authentication method:
    - If LLM_CLIENT_ID is set: Uses OAuth2 (Circuit internal LLM gateway)
    - Otherwise: Uses standard OpenAI API with OPENAI_API_KEY
    """
    # Check if we should use OAuth2 or standard OpenAI
    use_oauth2 = bool(os.environ.get("LLM_CLIENT_ID"))

    if use_oauth2:
        # Circuit (internal LLM gateway) with OAuth2
        from util import OAuth2TokenManager

        token_manager = OAuth2TokenManager()
        token = token_manager.get_token()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        base_url = OAuth2TokenManager.get_llm_base_url(model)

        print(f"🔑 Using OAuth2 authentication (Circuit): {base_url}")
        return OpenAI(
            base_url=base_url,
            api_key="placeholder",
            default_headers={"api-key": token},
        )
    else:
        # Standard OpenAI API
        print("🔑 Using standard OpenAI API key")
        return OpenAI()


def create_azure_embedding_client() -> AzureOpenAI | None:
    """Create Azure OpenAI client for embeddings if configured."""
    has_azure = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_API_VERSION")
    )

    if not has_azure:
        return None

    print(
        f"🔑 Using Azure OpenAI for embeddings: {os.environ.get('AZURE_OPENAI_ENDPOINT')}"
    )
    return AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )


# ============================================================================
# Mock Tools for Travel Assistant
# ============================================================================
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights."""
    random.seed(hash((origin, destination, date)) % (2**32))
    prices = [random.randint(200, 800) for _ in range(3)]
    airlines = ["United", "Delta", "Emirates"]
    times = ["06:00", "12:30", "18:45"]

    results = []
    for airline, price, time in zip(airlines, prices, times):
        results.append(f"  - {airline}: ${price}, departs {time}")

    return f"Flights from {origin} to {destination} on {date}:\n" + "\n".join(
        results
    )


def search_hotels(city: str, check_in: str, check_out: str) -> str:
    """Search for available hotels."""
    random.seed(hash((city, check_in)) % (2**32))

    hotels = [
        ("Grand Hotel", random.randint(150, 300), 4.5),
        ("Boutique Inn", random.randint(100, 200), 4.8),
        ("Budget Lodge", random.randint(50, 100), 3.9),
    ]

    results = []
    for name, price, rating in hotels:
        results.append(f"  - {name}: ${price}/night, {rating}★")

    return f"Hotels in {city} ({check_in} to {check_out}):\n" + "\n".join(
        results
    )


def search_activities(city: str) -> str:
    """Search for activities in a city."""
    activities_db = {
        "tokyo": [
            "🏯 Senso-ji Temple Tour - $25",
            "🍣 Tsukiji Market Food Tour - $75",
            "🎮 Akihabara Gaming District - Free",
            "🌸 Ueno Park & Museums - $15",
        ],
        "paris": [
            "🗼 Eiffel Tower Visit - $30",
            "🎨 Louvre Museum - $20",
            "🥐 Cooking Class - $90",
            "🚶 Montmartre Walking Tour - $35",
        ],
        "new york": [
            "🗽 Statue of Liberty - $25",
            "🎭 Broadway Show - $150",
            "🌳 Central Park Bike Tour - $40",
            "🏛️ Metropolitan Museum - $30",
        ],
    }

    city_lower = city.lower()
    if city_lower in activities_db:
        activities = activities_db[city_lower]
    else:
        activities = [
            "🏛️ City Tour - $35",
            "🍴 Local Food Experience - $50",
            "🎭 Cultural Show - $45",
        ]

    return f"Activities in {city}:\n" + "\n".join(
        f"  - {a}" for a in activities
    )


def get_weather(city: str, date: str) -> str:
    """Get weather forecast for a city."""
    random.seed(hash((city, date)) % (2**32))
    temp = random.randint(15, 30)
    conditions = random.choice(
        ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"]
    )
    return f"Weather in {city} on {date}: {temp}°C, {conditions}"


# Tool registry
TOOLS: dict[str, Any] = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "search_activities": search_activities,
    "get_weather": get_weather,
}

# Tool definitions for OpenAI
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two cities on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure city",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Arrival city",
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date (YYYY-MM-DD)",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for available hotels in a city for given dates",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date (YYYY-MM-DD)",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date (YYYY-MM-DD)",
                    },
                },
                "required": ["city", "check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_activities",
            "description": "Search for activities and attractions in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast for a city on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "date": {
                        "type": "string",
                        "description": "Date (YYYY-MM-DD)",
                    },
                },
                "required": ["city", "date"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful travel assistant. Help users plan their trips by:
1. Searching for flights, hotels, and activities
2. Providing weather information
3. Making personalized recommendations

When a user asks about travel planning, use the available tools to gather information.
Be concise but helpful in your responses. Summarize the options clearly."""


def execute_tool_calls(tool_calls: list) -> list[dict]:
    """Execute tool calls and return results."""
    results = []
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"   🔧 Calling {func_name}({args})")

        if func_name in TOOLS:
            result = TOOLS[func_name](**args)
        else:
            result = f"Unknown tool: {func_name}"

        results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": result,
            }
        )
        print(f"   📋 Result: {result[:100]}...")

    return results


def _get_user_field() -> str | None:
    """Get user field for Circuit gateway (contains app key as JSON)."""
    app_key = os.environ.get("LLM_APP_KEY")
    if app_key:
        return json.dumps({"appkey": app_key})
    return None


def run_assistant(client: OpenAI, user_message: str, model: str) -> str:
    """Run the travel assistant with tool calling loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    print(f"\n💬 User: {user_message}")
    print("-" * 50)

    max_iterations = 5
    iteration = 0
    user_field = _get_user_field()

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Iteration {iteration}")

        kwargs = {
            "model": model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
        }
        if user_field:
            kwargs["user"] = user_field

        response = client.chat.completions.create(**kwargs)

        assistant_message = response.choices[0].message

        # Check if there are tool calls
        if assistant_message.tool_calls:
            print(f"   🛠️  {len(assistant_message.tool_calls)} tool call(s)")

            # Add assistant message with tool calls to history
            messages.append(assistant_message)

            # Execute tools and add results
            tool_results = execute_tool_calls(assistant_message.tool_calls)
            messages.extend(tool_results)
        else:
            # No more tool calls, we have the final response
            final_response = assistant_message.content
            print(f"\n✅ Assistant: {final_response}")
            return final_response

    return "Max iterations reached. Please try a simpler query."


# ============================================================================
# Embeddings: Semantic Destination Search
# ============================================================================
DESTINATIONS_DB = [
    {
        "name": "Tokyo, Japan",
        "description": "Vibrant metropolis blending ancient temples with cutting-edge technology. Famous for sushi, anime, and cherry blossoms.",
        "tags": ["technology", "temples", "food", "culture", "shopping"],
    },
    {
        "name": "Paris, France",
        "description": "City of lights and love. Home to the Eiffel Tower, world-class museums, and exquisite cuisine.",
        "tags": ["romance", "art", "museums", "food", "architecture"],
    },
    {
        "name": "Bali, Indonesia",
        "description": "Tropical paradise with stunning beaches, rice terraces, and spiritual temples. Perfect for relaxation.",
        "tags": ["beach", "relaxation", "temples", "nature", "wellness"],
    },
    {
        "name": "New York City, USA",
        "description": "The city that never sleeps. Broadway shows, iconic skyline, diverse food scene, and world-famous museums.",
        "tags": ["entertainment", "food", "museums", "shopping", "nightlife"],
    },
    {
        "name": "Reykjavik, Iceland",
        "description": "Gateway to dramatic landscapes, northern lights, geysers, and glaciers. Adventure and natural wonders.",
        "tags": ["adventure", "nature", "northern lights", "hiking", "unique"],
    },
    {
        "name": "Marrakech, Morocco",
        "description": "Exotic bazaars, vibrant souks, and beautiful riads. A sensory experience of colors, spices, and history.",
        "tags": ["culture", "markets", "history", "food", "architecture"],
    },
]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def search_destinations_with_embeddings(
    client: OpenAI | AzureOpenAI,
    query: str,
    embedding_model: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Search destinations using semantic similarity with embeddings.

    This demonstrates:
    1. Generating embeddings for a search query (single)
    2. Generating embeddings for destination descriptions (batch)
    3. Finding most similar destinations using cosine similarity

    Works with both OpenAI and Azure OpenAI clients.
    """
    print(f"\n🔍 Semantic Search: '{query}'")
    print("-" * 50)

    # Get embedding for the search query
    print("   📊 Generating query embedding...")
    query_response = client.embeddings.create(
        model=embedding_model,
        input=query,
    )
    query_embedding = query_response.data[0].embedding
    print(f"   ✅ Query embedding: {len(query_embedding)} dimensions")

    # Get embeddings for all destinations (batch request)
    descriptions = [
        f"{d['name']}: {d['description']} Tags: {', '.join(d['tags'])}"
        for d in DESTINATIONS_DB
    ]

    print(
        f"   📊 Generating embeddings for {len(descriptions)} destinations..."
    )
    dest_response = client.embeddings.create(
        model=embedding_model,
        input=descriptions,
    )
    print("   ✅ Destination embeddings generated")

    # Calculate similarities
    similarities = []
    for i, dest in enumerate(DESTINATIONS_DB):
        dest_embedding = dest_response.data[i].embedding
        similarity = cosine_similarity(query_embedding, dest_embedding)
        similarities.append(
            {
                "destination": dest,
                "similarity": similarity,
            }
        )

    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:top_k]

    print(f"\n   🎯 Top {top_k} matches:")
    for i, result in enumerate(top_results, 1):
        dest = result["destination"]
        score = result["similarity"]
        print(f"      {i}. {dest['name']} (similarity: {score:.3f})")
        print(f"         {dest['description'][:60]}...")

    return top_results


# ============================================================================
# Main
# ============================================================================
def main():
    # Setup telemetry FIRST - this MUST be inside main() to avoid
    # multiprocessing spawn issues with evaluation worker process
    global trace_provider, tracer, logger_provider, OTLP_ENDPOINT
    trace_provider, tracer, logger_provider, OTLP_ENDPOINT = setup_telemetry()

    from opentelemetry import metrics

    print("=" * 70)
    print("✈️  Travel Assistant (OpenAI SDK + Tool Calling + Embeddings)")
    print("=" * 70)

    # Validate environment
    use_oauth2 = bool(os.environ.get("LLM_CLIENT_ID"))
    if use_oauth2:
        required = ["LLM_CLIENT_ID", "LLM_CLIENT_SECRET"]
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            print(f"\n❌ Missing OAuth2 credentials: {', '.join(missing)}")
            sys.exit(1)
    elif not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ Missing OPENAI_API_KEY (or set LLM_CLIENT_ID for OAuth2)")
        sys.exit(1)

    # Create clients
    chat_client = create_openai_client()
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # For embeddings: Azure is required (Circuit chat endpoint doesn't support embeddings)
    embedding_client = create_azure_embedding_client()
    embedding_model = os.environ.get(
        "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
    )

    # =========================================================================
    # Part 1: Semantic Destination Search with Embeddings (requires Azure)
    # =========================================================================
    if embedding_client:
        print("\n" + "=" * 70)
        print("📍 PART 1: Semantic Destination Search (Azure Embeddings)")
        print("=" * 70)

        search_queries = [
            "I want a relaxing beach vacation with temples",
            "Looking for adventure and unique natural phenomena",
            "Romantic city with great art and food",
        ]

        with tracer.start_as_current_span(
            name="POST /destinations/search",
            kind=SpanKind.SERVER,
            attributes={
                "http.request.method": "POST",
                "http.route": "/destinations/search",
            },
        ):
            for query in search_queries:
                search_destinations_with_embeddings(
                    embedding_client, query, embedding_model
                )
    else:
        print("\n" + "=" * 70)
        print("⏭️  SKIPPING: Semantic Destination Search (Embeddings)")
        print("   Azure OpenAI not configured. Set these env vars to enable:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_API_VERSION")
        print("   - AZURE_EMBEDDING_DEPLOYMENT")
        print("=" * 70)

    # =========================================================================
    # Part 2: Travel Planning with Tool Calling
    # =========================================================================
    print("\n" + "=" * 70)
    print("🗺️  PART 2: Travel Planning (Chat + Tool Calling)")
    print("=" * 70)

    # Calculate dates
    departure = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    return_date = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")

    planning_query = (
        f"I want to plan a trip to Tokyo from San Francisco "
        f"from {departure} to {return_date}. "
        "Find me flights, hotels, and suggest some activities. "
        "Also, what's the weather like?"
    )

    # Use TelemetryHandler workflow to wrap all OpenAI calls in a parent span
    # This ensures evaluation spans are properly correlated
    try:
        from opentelemetry.util.genai.handler import get_telemetry_handler
        from opentelemetry.util.genai.types import Workflow

        handler = get_telemetry_handler()
        if handler is not None:
            # Create and start a workflow to act as parent for all OpenAI calls
            workflow = Workflow(
                name="travel_planning",
                workflow_type="agent",
                description="Travel planning assistant with tool calling",
            )
            handler.start_workflow(workflow)

            try:
                with tracer.start_as_current_span(
                    name="POST /travel/plan",
                    kind=SpanKind.SERVER,
                    attributes={
                        "http.request.method": "POST",
                        "http.route": "/travel/plan",
                    },
                ):
                    run_assistant(chat_client, planning_query, model)
            finally:
                handler.stop_workflow(workflow)
        else:
            # No handler available, just run without workflow wrapper
            with tracer.start_as_current_span(
                name="POST /travel/plan",
                kind=SpanKind.SERVER,
                attributes={
                    "http.request.method": "POST",
                    "http.route": "/travel/plan",
                },
            ):
                run_assistant(chat_client, planning_query, model)
    except ImportError:
        # util-genai not available, just run with tracer span
        with tracer.start_as_current_span(
            name="POST /travel/plan",
            kind=SpanKind.SERVER,
            attributes={
                "http.request.method": "POST",
                "http.route": "/travel/plan",
            },
        ):
            run_assistant(chat_client, planning_query, model)

    # Wait for async evaluations to complete
    print("\n" + "=" * 70)
    print("⏳ Waiting for evaluations to complete...")
    try:
        from opentelemetry.util.genai.handler import get_telemetry_handler

        handler = get_telemetry_handler()
        if handler is not None:
            handler.wait_for_evaluations(timeout=60.0)
            print("✅ Evaluations completed!")
    except Exception as e:
        print(f"⚠️ Evaluation flush error: {e}")

    # Flush all telemetry
    print(f"📊 Flushing telemetry to Console + OTLP ({OTLP_ENDPOINT})...")
    trace_provider.force_flush()
    metrics.get_meter_provider().force_flush()
    logger_provider.force_flush()
    print("✅ Traces, Metrics, and Logs exported!")
    print("=" * 70)


if __name__ == "__main__":
    # This guard is REQUIRED for multiprocessing spawn mode (macOS default)
    # Without it, the evaluation worker process would re-run OTel setup
    main()
