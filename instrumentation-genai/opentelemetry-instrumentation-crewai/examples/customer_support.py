from crewai import Agent, Task, Crew

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
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

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

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

# Enable metrics in genai-util (defaults to span-only)
os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric"

support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful "
        "support representative in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        " are now working on providing "
		"support to {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
	allow_delegation=False,
	verbose=False
)

# By not setting allow_delegation=False, allow_delegation takes its default value of being True.
# This means the agent can delegate its work to another agent which is better suited to do a particular task.


support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the "
    "best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the support representative "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=False
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/en/concepts/crews"
)

# You are passing the Tool on the Task Level
inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)

# quality_assurance_review is not using any Tool(s)
# Here the QA Agent will only review the work of the Support Agent
quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
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
  memory=True
)

inputs = {
    "customer": "Splunk Olly for AI",
    "person": "Aditya Mehra",
    "inquiry": "I need help with setting up a Crew "
               "and kicking it off, specifically "
               "how can I add memory to my crew? "
               "Can you provide guidance?"
}

OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider)
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
