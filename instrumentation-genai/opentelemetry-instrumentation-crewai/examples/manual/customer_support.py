"""
CrewAI Customer Support Example with Cisco LLM Integration.

This example demonstrates:
- Using Cisco Chat AI via LiteLLM with OAuth2 authentication
- Manual OpenTelemetry instrumentation for CrewAI
- Proper telemetry flushing for traces and metrics

Environment Variables:
    CISCO_CLIENT_ID: Your Cisco OAuth2 client ID
    CISCO_CLIENT_SECRET: Your Cisco OAuth2 client secret
    CISCO_APP_KEY: Your Cisco app key
    OTEL_CONSOLE_OUTPUT: Set to "true" for local debugging
"""

from crewai import Agent, Task, Crew, LLM

import sys
import time
from crewai_tools import ScrapeWebsiteTool

import os
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from util import CiscoTokenManager

# Enable console output for local debugging (set to "false" in cluster)
ENABLE_CONSOLE_OUTPUT = os.environ.get("OTEL_CONSOLE_OUTPUT", "false").lower() == "true"

# Configure Trace Provider with OTLP exporter
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

if ENABLE_CONSOLE_OUTPUT:
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# CRITICAL: Register the tracer provider globally so it can be flushed
trace.set_tracer_provider(tracer_provider)

# Configure Metrics Provider with OTLP exporter
metric_readers = [
    PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=60000  # Export every 60 seconds for production
    )
]

if ENABLE_CONSOLE_OUTPUT:
    metric_readers.append(
        PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=60000
        )
    )

meter_provider = metrics_sdk.MeterProvider(metric_readers=metric_readers)
metrics.set_meter_provider(meter_provider)

# =============================================================================
# LLM Configuration - Cisco Chat AI
# =============================================================================

# Cisco API requires an appkey in the 'user' field of the request body
# Get this from the Cisco API portal
CISCO_APP_KEY = os.environ.get("CISCO_APP_KEY")

# Initialize token manager (uses CISCO_CLIENT_ID, CISCO_CLIENT_SECRET env vars)
token_manager = CiscoTokenManager()

def get_cisco_llm():
    """Create LLM instance with fresh token for Cisco Chat AI."""
    import json
    token = token_manager.get_token()
    
    # Cisco requires:
    # 1. api-key header with OAuth token
    # 2. user field in request body with JSON-encoded appkey
    return LLM(
        model="openai/gpt-4o-mini",
        base_url=CiscoTokenManager.get_llm_base_url("gpt-4o-mini"),
        api_key="placeholder",  # Required by LiteLLM but Cisco uses api-key header
        extra_headers={
            "api-key": token,  # Cisco expects OAuth token in api-key header
        },
        # Pass appkey in user field as JSON string (required by Cisco)
        user=json.dumps({"appkey": CISCO_APP_KEY}),
        temperature=0.7,
    )

cisco_llm = get_cisco_llm()


# =============================================================================
# CrewAI Agents
# =============================================================================

support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful support representative in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working on providing "
        "support to {customer}, a super important customer for your company. "
        "You need to make sure that you provide the best support! "
        "Make sure to provide full complete answers, and make no assumptions."
    ),
    llm=cisco_llm,
    allow_delegation=False,
    verbose=False,
    cache=False,  # Disable agent caching to avoid embedding calls
)

support_quality_assurance_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Get recognition for providing the best support quality assurance in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working with your team "
        "on a request from {customer} ensuring that the support representative is "
        "providing the best support possible. "
        "You need to make sure that the support representative is providing full "
        "complete answers, and make no assumptions."
    ),
    llm=cisco_llm,
    verbose=False,
    cache=False,  # Disable agent caching to avoid embedding calls
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/en/concepts/crews"
)

inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know to provide the best support possible. "
        "You must strive to provide a complete and accurate response to the customer's inquiry."
    ),
    expected_output=(
        "A detailed, informative response to the customer's inquiry that addresses "
        "all aspects of their question. "
        "The response should include references to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, leaving no questions unanswered, "
        "and maintain a helpful and friendly tone throughout."
    ),
    tools=[docs_scrape_tool],
    agent=support_agent,
)

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
        "high-quality standards expected for customer support. "
        "Verify that all parts of the customer's inquiry have been addressed "
        "thoroughly, with a helpful and friendly tone. "
        "Check for references and sources used to find the information, "
        "ensuring the response is well-supported and leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response ready to be sent to the customer. "
        "This response should fully address the customer's inquiry, incorporating all "
        "relevant feedback and improvements. "
        "Don't be too formal, we are a chill and cool company "
        "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

# Setting memory=True when putting the crew together enables Memory
crew = Crew(
    agents=[support_agent, support_quality_assurance_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=False,
    memory=False
)

inputs = {
    "customer": "Splunk Olly for AI",
    "person": "Aditya Mehra",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}

CrewAIInstrumentor().instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider
)

def flush_telemetry():
    """Flush all OpenTelemetry providers before exit to ensure traces and metrics are exported."""
    print("\n[FLUSH] Starting telemetry flush", flush=True)
    
    # Flush traces
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            print("[FLUSH] Flushing traces (timeout=30s)", flush=True)
            tracer_provider.force_flush(timeout_millis=30000)
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush traces: {e}", flush=True)
    
    # Flush metrics
    try:
        meter_provider_instance = metrics.get_meter_provider()
        if hasattr(meter_provider_instance, "force_flush"):
            print("[FLUSH] Flushing metrics (timeout=30s)", flush=True)
            meter_provider_instance.force_flush(timeout_millis=30000)
        if hasattr(meter_provider_instance, "shutdown"):
            print("[FLUSH] Shutting down metrics provider", flush=True)
            meter_provider_instance.shutdown()
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush metrics: {e}", flush=True)
    
    # Give batch processors time to complete final export
    time.sleep(2)
    print("[FLUSH] Telemetry flush complete\n", flush=True)

if __name__ == "__main__":
    exit_code = 0
    try:
        # Refresh token and recreate LLM with fresh token
        fresh_token = token_manager.get_token()
        print(f"[AUTH] Token obtained (length: {len(fresh_token)})")
        
        # Recreate LLM with fresh token in headers
        cisco_llm = get_cisco_llm()
        
        # Update agents with fresh LLM
        support_agent.llm = cisco_llm
        support_quality_assurance_agent.llm = cisco_llm
        
        result = crew.kickoff(inputs=inputs)
        print("\n[SUCCESS] Crew execution completed")
    except Exception as e:
        print(f"\n[ERROR] Crew execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # CRITICAL: Always flush telemetry to ensure spans and metrics are exported
        print("\n" + "="*100)
        print("METRICS OUTPUT BELOW - Look for gen_ai.agent.duration and gen_ai.workflow.duration")
        print("="*100 + "\n")
        flush_telemetry()
        sys.exit(exit_code)
