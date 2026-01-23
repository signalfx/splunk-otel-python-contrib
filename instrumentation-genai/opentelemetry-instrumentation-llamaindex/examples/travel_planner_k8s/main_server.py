"""
Travel Planner Server using LlamaIndex ReActAgent.

This server exposes an HTTP endpoint for travel planning requests and uses
OpenTelemetry instrumentation to capture traces and metrics.
"""

import os

import asyncio
import base64
import json
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import requests
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor


# OAuth2 Token Manager for CircuIT
class OAuth2TokenManager:
    """Manages OAuth2 token lifecycle for CircuIT.

    Handles token retrieval, file-based caching, and automatic refresh with a 5-minute buffer.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str = None,
        cache_file: str = "/tmp/circuit_token_cache.json",
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.cache_file = cache_file

    def _get_cached_token(self):
        """Get token from cache file if valid."""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data["access_token"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _fetch_new_token(self):
        """Request a new access token from the OAuth2 server."""
        # Create Basic Auth header
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}
        if self.scope:
            data["scope"] = self.scope

        response = requests.post(self.token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        # Cache token to file with secure permissions
        cache_data = {
            "access_token": token_data["access_token"],
            "expires_at": expires_at.isoformat(),
        }

        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        os.chmod(self.cache_file, 0o600)  # rw------- (owner only)

        return token_data["access_token"]

    def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()

    def cleanup_token_cache(self):
        """Securely remove token cache file."""
        if os.path.exists(self.cache_file):
            # Overwrite file with zeros before deletion for security
            with open(self.cache_file, "r+b") as f:
                length = f.seek(0, 2)  # Get file size
                f.seek(0)
                f.write(b"\0" * length)  # Overwrite with zeros
            os.remove(self.cache_file)


# Custom LLM for CircuIT
class CircuITLLM(CustomLLM):
    """Custom LLM implementation for Cisco CircuIT OAuth2 gateway."""

    api_url: str
    token_manager: OAuth2TokenManager
    app_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=128000,
            num_output=4096,
        )

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Send chat request to CircuIT."""
        access_token = self.token_manager.get_token()

        # Convert LlamaIndex ChatMessage to OpenAI format
        api_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        # CircuIT requires appkey in user field as JSON string
        user_field = json.dumps({"appkey": self.app_key})

        payload = {
            "messages": api_messages,
            "temperature": self.temperature,
            "user": user_field,
        }

        headers = {
            "api-key": access_token,
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            raw=result,
        )

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat (uses sync implementation for now)."""
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: list[ChatMessage], **kwargs: Any):
        """Stream chat (non-streaming implementation - yields single response)."""
        # CircuIT doesn't support streaming, so just yield the complete response
        response = self.chat(messages, **kwargs)
        yield response

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete is not used by ReActAgent, but required by interface."""
        raise NotImplementedError("Use chat() instead")

    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream complete not implemented."""
        raise NotImplementedError("Use chat() instead")


# LLM Configuration
_llm_instance = None


def get_llm():
    """Get or create LLM instance (OpenAI or CircuIT based on env vars)."""
    global _llm_instance

    if _llm_instance is None:
        # Check for CircuIT credentials first
        circuit_vars = {
            "CIRCUIT_BASE_URL": os.getenv("CIRCUIT_BASE_URL"),
            "CIRCUIT_TOKEN_URL": os.getenv("CIRCUIT_TOKEN_URL"),
            "CIRCUIT_CLIENT_ID": os.getenv("CIRCUIT_CLIENT_ID"),
            "CIRCUIT_CLIENT_SECRET": os.getenv("CIRCUIT_CLIENT_SECRET"),
            "CIRCUIT_APP_KEY": os.getenv("CIRCUIT_APP_KEY"),
        }

        # Check if all CircuIT vars are set and not empty
        has_circuit = all(
            val is not None and val.strip() != "" for val in circuit_vars.values()
        )

        if has_circuit:
            print("✓ Using CircuIT LLM")
            print(f"  Base URL: {circuit_vars['CIRCUIT_BASE_URL']}")
            token_manager = OAuth2TokenManager(
                token_url=circuit_vars["CIRCUIT_TOKEN_URL"],
                client_id=circuit_vars["CIRCUIT_CLIENT_ID"],
                client_secret=circuit_vars["CIRCUIT_CLIENT_SECRET"],
                scope=os.getenv("CIRCUIT_SCOPE"),
            )
            _llm_instance = CircuITLLM(
                api_url=circuit_vars["CIRCUIT_BASE_URL"],
                token_manager=token_manager,
                app_key=circuit_vars["CIRCUIT_APP_KEY"],
                model_name="gpt-4o-mini",
                temperature=0,
            )
        elif os.getenv("OPENAI_API_KEY"):
            print("✓ Using OpenAI LLM")
            _llm_instance = OpenAI(model="gpt-4o-mini", temperature=0)
        else:
            # Debug: show which vars are missing
            missing = [
                name
                for name, val in circuit_vars.items()
                if not val or val.strip() == ""
            ]
            error_msg = "No LLM credentials found.\n"
            if missing:
                error_msg += f"Missing CircuIT variables: {', '.join(missing)}\n"
            error_msg += "Set either OPENAI_API_KEY or all CIRCUIT_* variables."
            raise ValueError(error_msg)

    return _llm_instance


# Setup Telemetry
def setup_telemetry():
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))


# Define Travel Planning Tools
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between two cities on a specific date."""
    print(f"  [Tool] Searching flights from {origin} to {destination} on {date}...")

    # Simulate flight search results
    flight_price = 800
    return (
        f"Found Flight UA{abs(hash(origin + destination)) % 1000}: "
        f"{origin} → {destination} on {date}, "
        f"Price: ${flight_price}, "
        f"Departure: 10:00 AM, Arrival: 2:00 PM"
    )


def search_hotels(city: str, check_in: str, check_out: str) -> str:
    """Search for hotels in a city for given check-in and check-out dates."""
    print(f"  [Tool] Searching hotels in {city} from {check_in} to {check_out}...")

    # Simulate hotel search results
    nightly_rate = 200
    return (
        f"Found Hotel Grand {city}: "
        f"Available from {check_in} to {check_out}, "
        f"Rate: ${nightly_rate}/night, "
        f"Rating: 4.5/5, Amenities: WiFi, Breakfast, Pool"
    )


def search_activities(city: str) -> str:
    """Search for activities and attractions in a city."""
    print(f"  [Tool] Searching activities in {city}...")

    activities = [
        f"City Tour of {city} - $50",
        f"Food Tour in {city} - $80",
        f"Museum Pass for {city} - $40",
    ]

    return f"Recommended activities: {', '.join(activities)}"


# Global agent instances
_flight_agent = None
_hotel_agent = None
_activity_agent = None


def get_flight_agent():
    """Get or create the flight search agent."""
    global _flight_agent
    if _flight_agent is None:
        llm = get_llm()
        tools = [FunctionTool.from_defaults(fn=search_flights)]
        system_prompt = "You are a flight search specialist. Use the search_flights tool to find flights, then provide the result."
        _flight_agent = ReActAgent(
            tools=tools, llm=llm, verbose=True, system_prompt=system_prompt
        )
    return _flight_agent


def get_hotel_agent():
    """Get or create the hotel search agent."""
    global _hotel_agent
    if _hotel_agent is None:
        llm = get_llm()
        tools = [FunctionTool.from_defaults(fn=search_hotels)]
        system_prompt = "You are a hotel search specialist. Use the search_hotels tool to find hotels, then provide the result."
        _hotel_agent = ReActAgent(
            tools=tools, llm=llm, verbose=True, system_prompt=system_prompt
        )
    return _hotel_agent


def get_activity_agent():
    """Get or create the activity search agent."""
    global _activity_agent
    if _activity_agent is None:
        llm = get_llm()
        tools = [FunctionTool.from_defaults(fn=search_activities)]
        system_prompt = "You are an activity recommendation specialist. Use the search_activities tool to find activities, then provide the result."
        _activity_agent = ReActAgent(
            tools=tools, llm=llm, verbose=True, system_prompt=system_prompt
        )
    return _activity_agent


# Multi-Agent Workflow Orchestrator
async def invoke_workflow(
    origin: str,
    destination: str,
    departure_date: str,
    check_out_date: str,
    budget: int = 3000,
    duration: int = 5,
    travelers: int = 2,
    interests: list = None,
) -> str:
    """
    Orchestrates the multi-agent travel planning workflow.

    Workflow: invoke_workflow → flight_specialist → hotel_specialist → activity_specialist

    Args:
        origin: Departure city
        destination: Arrival city
        departure_date: Departure date (YYYY-MM-DD)
        check_out_date: Hotel check-out date (YYYY-MM-DD)
        budget: Total budget in USD
        duration: Trip duration in days
        travelers: Number of travelers
        interests: List of traveler interests

    Returns:
        Aggregated results from all specialist agents
    """
    results = []

    # 1. Flight Specialist Agent
    print("\n--- Flight Specialist Agent ---")
    flight_agent = get_flight_agent()
    flight_query = f"Search for flights from {origin} to {destination} departing on {departure_date}"
    flight_handler = flight_agent.run(user_msg=flight_query, max_iterations=3)
    flight_response = await flight_handler
    results.append(f"Flights: {flight_response}")

    # 2. Hotel Specialist Agent
    print("\n--- Hotel Specialist Agent ---")
    hotel_agent = get_hotel_agent()
    hotel_query = (
        f"Search for hotels in {destination} from {departure_date} to {check_out_date}"
    )
    hotel_handler = hotel_agent.run(user_msg=hotel_query, max_iterations=3)
    hotel_response = await hotel_handler
    results.append(f"Hotels: {hotel_response}")

    # 3. Activity Specialist Agent
    print("\n--- Activity Specialist Agent ---")
    activity_agent = get_activity_agent()
    activity_query = f"Recommend activities in {destination}"
    activity_handler = activity_agent.run(user_msg=activity_query, max_iterations=3)
    activity_response = await activity_handler
    results.append(f"Activities: {activity_response}")

    return "\n\n".join(results)


class TravelPlannerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for travel planning."""

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests for travel planning."""
        if self.path == "/plan":
            # Create a root span for the HTTP request
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                "POST /plan",
                kind=trace.SpanKind.SERVER,
                attributes={
                    "http.method": "POST",
                    "http.target": "/plan",
                    "http.scheme": "http",
                },
            ) as span:
                try:
                    content_length = int(self.headers["Content-Length"])
                    post_data = self.rfile.read(content_length)
                    request_data = json.loads(post_data.decode("utf-8"))

                    # Extract parameters
                    destination = request_data.get("destination", "Paris")
                    origin = request_data.get("origin", "New York")
                    budget = request_data.get("budget", 3000)
                    duration = request_data.get("duration", 5)
                    travelers = request_data.get("travelers", 2)
                    interests = request_data.get("interests", ["sightseeing", "food"])
                    departure_date = request_data.get("departure_date", "2024-06-01")

                    print(f"\n{'=' * 60}")
                    print("New Travel Planning Request")
                    print(f"{'=' * 60}")
                    print(f"Destination: {destination}")
                    print(f"Origin: {origin}")
                    print(f"Budget: ${budget}")
                    print(f"Duration: {duration} days")
                    print(f"Travelers: {travelers}")
                    print(f"Interests: {', '.join(interests)}")
                    print(f"{'=' * 60}\n")

                    # Calculate check-out date
                    check_in = datetime.strptime(departure_date, "%Y-%m-%d")
                    check_out = check_in + timedelta(days=duration)
                    check_out_date = check_out.strftime("%Y-%m-%d")

                    # Invoke the multi-agent workflow
                    try:
                        asyncio.run(
                            invoke_workflow(
                                origin=origin,
                                destination=destination,
                                departure_date=departure_date,
                                check_out_date=check_out_date,
                                budget=budget,
                                duration=duration,
                                travelers=travelers,
                                interests=interests,
                            )
                        )
                    except RuntimeError as e:
                        if (
                            "asyncio.run() cannot be called from a running event loop"
                            in str(e)
                        ):
                            # Fallback for when event loop is already running
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(
                                    invoke_workflow(
                                        origin=origin,
                                        destination=destination,
                                        departure_date=departure_date,
                                        check_out_date=check_out_date,
                                        budget=budget,
                                        duration=duration,
                                        travelers=travelers,
                                        interests=interests,
                                    )
                                )
                            finally:
                                loop.close()
                        else:
                            raise

                    print(f"\n{'=' * 60}")
                    print("Planning Complete")
                    print(f"{'=' * 60}\n")

                    span.set_attribute("http.status_code", 200)

                except Exception as e:
                    print(f"Error processing request: {e}")
                    import traceback

                    traceback.print_exc()

                    span.set_attribute("http.status_code", 500)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))

                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"status": "error", "error": str(e)}).encode()
                    )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"{self.address_string()} - {format % args}")


def main():
    """Start the travel planner server."""
    # Setup telemetry first
    setup_telemetry()

    # Auto-instrument LlamaIndex - pass meter_provider to enable metrics
    tracer_provider = trace.get_tracer_provider()
    meter_provider = metrics.get_meter_provider()
    LlamaindexInstrumentor().instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )

    # Try to initialize LLM - this will fail with a clear error if credentials are missing
    try:
        get_llm()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Start HTTP server
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), TravelPlannerHandler)

    print(f"\n{'=' * 60}")
    print("Travel Planner Server Starting")
    print(f"{'=' * 60}")
    print(f"Port: {port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"Planning endpoint: POST http://localhost:{port}/plan")
    print(f"{'=' * 60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "shutdown"):
            meter_provider.shutdown()

    return 0


if __name__ == "__main__":
    exit(main())
