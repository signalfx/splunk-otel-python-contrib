"""
CrewAI Customer Support Example with OpenTelemetry Instrumentation.

This example demonstrates:
- A simple 2-agent customer support crew (Support Rep + QA Specialist)
- OAuth2 LLM authentication via LiteLLM
- Both manual and zero-code OpenTelemetry instrumentation modes

Instrumentation Modes:
    Manual mode (default):     python customer_support.py
    Zero-code mode:            opentelemetry-instrument python customer_support.py

Environment Variables:
    OTEL_MANUAL_INSTRUMENTATION: Set to "false" for zero-code mode (default: "true")
    LLM_CLIENT_ID: OAuth2 client ID
    LLM_CLIENT_SECRET: OAuth2 client secret
    LLM_TOKEN_URL: OAuth2 token endpoint
    LLM_BASE_URL: LLM endpoint base URL
    LLM_APP_KEY: App key for request tracking (optional)
    EVAL_FLUSH_WAIT_SECONDS: Time to wait for evaluations (default: 60)
    OTEL_CONSOLE_OUTPUT: Enable console output for debugging (default: false)
"""

import json
import os
import sys
import time
from typing import Callable

# =============================================================================
# Environment Defaults
# =============================================================================

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event,splunk")

# =============================================================================
# Instrumentation Mode Detection
# =============================================================================

MANUAL_INSTRUMENTATION = (
    os.environ.get("OTEL_MANUAL_INSTRUMENTATION", "true").lower() == "true"
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
FLUSH_TIMEOUT_MS = 30000
COMPANY_URL = "https://crewai.com"
DOCS_URL = "https://docs.crewai.com/"


# =============================================================================
# Utility Functions (DRY)
# =============================================================================


def log(tag: str, message: str, stderr: bool = False) -> None:
    """Unified logging with consistent format."""
    output = sys.stderr if stderr else sys.stdout
    print(f"[{tag}] {message}", file=output, flush=True)


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with fallback."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return get_env(key, str(default)).lower() == "true"


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    return int(get_env(key, str(default)))


def safe_flush(provider_getter: Callable, provider_name: str) -> None:
    """Safely flush an OpenTelemetry provider (DRY pattern)."""
    try:
        provider = provider_getter()
        if hasattr(provider, "force_flush"):
            log("FLUSH", f"Flushing {provider_name} (timeout=30s)")
            provider.force_flush(timeout_millis=FLUSH_TIMEOUT_MS)
        if hasattr(provider, "shutdown") and provider_name == "metrics":
            log("FLUSH", f"Shutting down {provider_name} provider")
            provider.shutdown()
    except Exception as e:
        log("FLUSH", f"Warning: Could not flush {provider_name}: {e}")


# =============================================================================
# OpenTelemetry Setup (Single Responsibility)
# =============================================================================


def setup_telemetry():
    """Configure OpenTelemetry providers for traces, metrics, and logs."""
    from opentelemetry import trace, metrics, _logs, _events
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk import metrics as metrics_sdk
    from opentelemetry.sdk._events import EventLoggerProvider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        BatchSpanProcessor,
    )
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter,
    )

    enable_console = get_env_bool("OTEL_CONSOLE_OUTPUT")

    # Traces
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if enable_console:
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_readers = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(), export_interval_millis=60000
        )
    ]
    if enable_console:
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=60000
            )
        )
    meter_provider = metrics_sdk.MeterProvider(metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)

    # Logs (required for evals)
    _logs.set_logger_provider(LoggerProvider())
    _logs.get_logger_provider().add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter())
    )

    # Events (required for evals)
    _events.set_event_logger_provider(EventLoggerProvider())

    return tracer_provider, meter_provider


def apply_instrumentation(tracer_provider, meter_provider) -> None:
    """Apply OpenTelemetry instrumentation to frameworks."""
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    log("INIT", "CrewAI instrumentation applied (traces, metrics, logs/events)")

    # OpenAI v2 instrumentation (uncomment if needed)
    # from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    # OpenAIInstrumentor().instrument(
    #     tracer_provider=tracer_provider, meter_provider=meter_provider
    # )
    # log("INIT", "OpenAI v2 instrumentation applied")


def flush_telemetry() -> None:
    """Flush all OpenTelemetry providers before exit."""
    if not MANUAL_INSTRUMENTATION:
        # Zero-code mode: only wait for evaluations, provider flush handled by opentelemetry-instrument
        eval_wait = get_env_int("EVAL_FLUSH_WAIT_SECONDS", 60)
        log("Telemetry", f"Waiting {eval_wait}s for evals to complete...", stderr=True)
        time.sleep(eval_wait)
        return

    from opentelemetry import trace, metrics, _logs

    log("FLUSH", "Starting telemetry flush")

    # Wait for background evaluation threads
    eval_wait = get_env_int("EVAL_FLUSH_WAIT_SECONDS", 60)
    log("Telemetry", f"Waiting {eval_wait}s for evals to complete...", stderr=True)
    time.sleep(eval_wait)

    # Flush all providers (DRY)
    safe_flush(trace.get_tracer_provider, "traces")
    safe_flush(metrics.get_meter_provider, "metrics")
    safe_flush(_logs.get_logger_provider, "logs")

    log("FLUSH", "Telemetry flush complete")


# =============================================================================
# LLM Configuration (Single Responsibility)
# =============================================================================


def create_llm_factory():
    """Create a factory for LLM instances with OAuth2 authentication."""
    from util import OAuth2TokenManager

    token_manager = OAuth2TokenManager()
    app_key = get_env("LLM_APP_KEY")

    def get_llm():
        """Create LLM instance with fresh OAuth2 token."""
        from crewai import LLM

        token = token_manager.get_token()
        llm_kwargs = {
            "model": f"openai/{DEFAULT_MODEL}",
            "base_url": OAuth2TokenManager.get_llm_base_url(DEFAULT_MODEL),
            "api_key": "placeholder",
            "extra_headers": {"api-key": token},
            "temperature": 0.7,
        }
        if app_key:
            llm_kwargs["user"] = json.dumps({"appkey": app_key})
        return LLM(**llm_kwargs)

    return get_llm


# =============================================================================
# Crew Definition (Single Responsibility)
# =============================================================================


def create_agents(llm, tools: list):
    """Create crew agents with shared configuration."""
    from crewai import Agent

    # Common agent config (DRY)
    common_config = {"llm": llm, "verbose": False, "cache": False}

    support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful support representative in your team",
        backstory=(
            f"You work at crewAI ({COMPANY_URL}) and are now working on providing "
            "support to {customer}, a super important customer for your company. "
            "You need to make sure that you provide the best support! "
            "Make sure to provide full complete answers, and make no assumptions. "
            "Use the documentation scraper tool to look up accurate information from "
            "the official CrewAI docs when needed."
        ),
        allow_delegation=False,
        tools=tools,
        **common_config,
    )

    qa_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the best support quality assurance in your team",
        backstory=(
            f"You work at crewAI ({COMPANY_URL}) and are now working with your team "
            "on a request from {customer} ensuring that the support representative is "
            "providing the best support possible. "
            "You need to make sure that the support representative is providing full "
            "complete answers, and make no assumptions."
        ),
        **common_config,
    )

    return support_agent, qa_agent


def create_tasks(support_agent, qa_agent):
    """Create crew tasks."""
    from crewai import Task

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
        agent=qa_agent,
    )

    return inquiry_resolution, quality_assurance_review


def create_crew(agents: list, tasks: list):
    """Create the crew with agents and tasks."""
    from crewai import Crew

    return Crew(agents=agents, tasks=tasks, verbose=False, memory=False)


def create_tools():
    """Create tools for the crew."""
    from crewai_tools import ScrapeWebsiteTool

    docs_scrape_tool = ScrapeWebsiteTool(website_url=DOCS_URL)
    log("TOOLS", "ScrapeWebsiteTool initialized for CrewAI docs")
    return [docs_scrape_tool]


# =============================================================================
# Main Execution
# =============================================================================


def main() -> int:
    """Main entry point for the customer support crew."""
    # Setup telemetry based on instrumentation mode
    if MANUAL_INSTRUMENTATION:
        log("INIT", "Manual instrumentation mode enabled")
        tracer_provider, meter_provider = setup_telemetry()
        apply_instrumentation(tracer_provider, meter_provider)
    else:
        log("INIT", "Zero-code instrumentation mode (use opentelemetry-instrument)")

    # Setup LLM
    get_llm = create_llm_factory()
    log("LLM", "Using OAuth2 authentication")

    # Create crew components
    tools = create_tools()
    llm = get_llm()
    support_agent, qa_agent = create_agents(llm, tools)
    inquiry_task, qa_task = create_tasks(support_agent, qa_agent)
    crew = create_crew([support_agent, qa_agent], [inquiry_task, qa_task])

    # Input data
    inputs = {
        "customer": "Splunk Observability for AI",
        "person": "Aditya Mehra",
        "inquiry": (
            "I need help with setting up a Crew and kicking it off, "
            "specifically how can I add memory to my crew? Can you provide guidance?"
        ),
    }

    exit_code = 0
    try:
        # Refresh LLM with fresh token before execution
        fresh_llm = get_llm()
        support_agent.llm = fresh_llm
        qa_agent.llm = fresh_llm
        log("AUTH", "OAuth2 token refreshed")

        log("START", "Executing Customer Support Crew")
        log("START", f"Customer: {inputs['customer']}")
        log("START", f"Person: {inputs['person']}")
        print("", flush=True)

        log("DEBUG", "Starting crew.kickoff()...")
        result = crew.kickoff(inputs=inputs)
        log("DEBUG", "crew.kickoff() completed")

        log("SUCCESS", "Crew execution completed")
        print("=" * 80, flush=True)
        print(result, flush=True)

    except Exception as e:
        log("ERROR", f"Crew execution failed: {e}", stderr=True)
        import traceback

        traceback.print_exc()
        exit_code = 1

    finally:
        print("\n" + "=" * 80, flush=True)
        print("TELEMETRY OUTPUT - Traces, Metrics, and Logs/Events", flush=True)
        print("=" * 80 + "\n", flush=True)
        flush_telemetry()
        log("EXIT", f"Exiting with code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())


# =============================================================================
# Instrumentation Modes:
# =============================================================================
# Manual mode (default):     python customer_support.py
# Zero-code mode:            opentelemetry-instrument python customer_support.py
#
# Set OTEL_MANUAL_INSTRUMENTATION=false to use zero-code mode
# =============================================================================
#
# Expected Trace Structure:
# =============================================================================
# gen_ai.workflow (Customer Support Crew)
# ├── gen_ai.step (Inquiry Resolution)
# │   └── invoke_agent (Senior Support Representative)
# │       ├── chat (OpenAI/LiteLLM)
# │       │   └── gen_ai.choice
# │       └── tool (Read website content - CrewAI docs)
# └── gen_ai.step (Quality Assurance Review)
#     └── invoke_agent (Support Quality Assurance Specialist)
#         └── chat (OpenAI/LiteLLM)
#             └── gen_ai.choice
# =============================================================================
