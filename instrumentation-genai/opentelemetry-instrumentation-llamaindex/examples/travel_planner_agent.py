import asyncio
import os
import sys

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.llamaindex import LlamaindexInstrumentor

# 1. Setup Telemetry
def setup_telemetry():
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(insecure=True))
    )

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(insecure=True))
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# 2. Define Tools
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between two cities on a specific date."""
    print(f"  [Tool] Searching flights from {origin} to {destination} on {date}...")
    return f"Flight UA123 from {origin} to {destination} on {date} costs $500."

def search_hotels(city: str, check_in: str) -> str:
    """Search for hotels in a city."""
    print(f"  [Tool] Searching hotels in {city} for {check_in}...")
    return f"Hotel Grand in {city} is available for $200/night."

def book_ticket(flight_number: str) -> str:
    """Book a flight ticket."""
    print(f"  [Tool] Booking flight {flight_number}...")
    return f"Confirmed booking for {flight_number}. Ticket #999."

# 3. Main Agent Logic
async def run_travel_planner():
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    setup_telemetry()
    
    # Instrument LlamaIndex
    LlamaindexInstrumentor().instrument()

    # Setup LLM
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

    # Create Tools
    tools = [
        FunctionTool.from_defaults(fn=search_flights),
        FunctionTool.from_defaults(fn=search_hotels),
        FunctionTool.from_defaults(fn=book_ticket),
    ]

    # Create Agent
    # ReActAgent in LlamaIndex uses the workflow engine internally
    agent = ReActAgent(tools=tools, llm=Settings.llm, verbose=True)

    # Run Workflow
    user_request = "I want to fly from New York to Paris on 2023-12-01. Find a flight and book it, then find a hotel."
    
    # We use the async run method which returns the handler we instrumented
    # This triggers wrap_agent_run -> WorkflowEventInstrumentor
    handler = agent.run(user_msg=user_request)
    response = await handler
    
    print(f"\nFinal Response: {response}")

    # Ensure spans are flushed before exit
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    if hasattr(provider, "shutdown"):
        provider.shutdown()

if __name__ == "__main__":
    asyncio.run(run_travel_planner())
