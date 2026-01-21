"""
CrewAI Financial Assistant Example with OpenTelemetry Instrumentation.

This example demonstrates:
- A multi-agent financial trading crew with Data Analyst, Strategy Developer,
  Trade Advisor, and Risk Advisor agents
- OAuth2 LLM authentication via LiteLLM
- Both manual and zero-code OpenTelemetry instrumentation modes

Instrumentation Modes:
    Manual mode (default):     python financial_assistant.py
    Zero-code mode:            opentelemetry-instrument python financial_assistant.py

Environment Variables:
    OTEL_MANUAL_INSTRUMENTATION: Set to "false" for zero-code mode (default: "true")

    # LLM Configuration (choose one):
    OPENAI_API_KEY: Direct OpenAI API key
    -- OR --
    LLM_CLIENT_ID: OAuth2 client ID
    LLM_CLIENT_SECRET: OAuth2 client secret
    LLM_TOKEN_URL: OAuth2 token endpoint
    LLM_BASE_URL: LLM endpoint base URL
    LLM_APP_KEY: App key for request tracking (optional)

    # OpenTelemetry Configuration:
    OTEL_SERVICE_NAME: Service name for telemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint
    OTEL_CONSOLE_OUTPUT: Set to "true" for local debugging
"""

import json
import os
import sys
import time

# Disable CrewAI's built-in telemetry
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric")

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# =============================================================================
# Instrumentation Mode Detection
# =============================================================================

MANUAL_INSTRUMENTATION = os.environ.get(
    "OTEL_MANUAL_INSTRUMENTATION", "true"
).lower() == "true"

# For console output in manual mode (for local debugging)
ENABLE_CONSOLE_OUTPUT = os.environ.get(
    "OTEL_CONSOLE_OUTPUT", "false"
).lower() == "true"

# =============================================================================
# Manual OpenTelemetry Setup (only when OTEL_MANUAL_INSTRUMENTATION=true)
# =============================================================================

tracer_provider = None
meter_provider = None

if MANUAL_INSTRUMENTATION:
    print("[INIT] Manual instrumentation mode enabled")
    
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk import metrics as metrics_sdk
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        BatchSpanProcessor,
    )
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter,
    )
    from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

    # Configure Trace Provider with OTLP exporter
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    if ENABLE_CONSOLE_OUTPUT:
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Register the tracer provider globally
    trace.set_tracer_provider(tracer_provider)

    # Configure Metrics Provider with OTLP exporter
    metric_readers = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=60000,  # Export every 60 seconds
        )
    ]

    if ENABLE_CONSOLE_OUTPUT:
        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=60000
            )
        )

    meter_provider = metrics_sdk.MeterProvider(metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)

    # Apply CrewAI instrumentation
    CrewAIInstrumentor().instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    print("[INIT] CrewAI instrumentation applied")
else:
    print("[INIT] Zero-code instrumentation mode (use opentelemetry-instrument)")


# =============================================================================
# LLM Configuration - OAuth2 or OpenAI
# =============================================================================

# Check if using OAuth2 or direct OpenAI
USE_OAUTH2 = bool(
    os.environ.get("LLM_CLIENT_ID") or os.environ.get("LLM_CLIENT_SECRET")
)

if USE_OAUTH2:
    from util import OAuth2TokenManager
    
    LLM_APP_KEY = os.environ.get("LLM_APP_KEY") or os.environ.get("CISCO_APP_KEY")
    token_manager = OAuth2TokenManager()

    def get_llm():
        """Create LLM instance with fresh OAuth2 token."""
        token = token_manager.get_token()

        llm_kwargs = {
            "model": "openai/gpt-4o-mini",
            "base_url": OAuth2TokenManager.get_llm_base_url("gpt-4o-mini"),
            "api_key": "placeholder",  # Required by LiteLLM but we use api-key header
            "extra_headers": {
                "api-key": token,  # OAuth token in api-key header
            },
            "temperature": 0.7,
        }

        if LLM_APP_KEY:
            llm_kwargs["user"] = json.dumps({"appkey": LLM_APP_KEY})

        return LLM(**llm_kwargs)

    print("[LLM] Using OAuth2 authentication")
else:
    def get_llm():
        """Create LLM instance with OpenAI API key."""
        return LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        )

    print("[LLM] Using OpenAI API key")


# =============================================================================
# Tools
# =============================================================================

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# =============================================================================
# CrewAI Agents
# =============================================================================

llm = get_llm()

data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time "
    "to identify trends and predict market movements.",
    backstory=(
        "Specializing in financial markets, this agent "
        "uses statistical modeling and machine learning "
        "to provide crucial insights. With a knack for data, "
        "the Data Analyst Agent is the cornerstone for "
        "informing trading decisions."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    cache=False,  # Disable caching to avoid embedding calls
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based "
    "on insights from the Data Analyst Agent.",
    backstory=(
        "Equipped with a deep understanding of financial "
        "markets and quantitative analysis, this agent "
        "devises and refines trading strategies. It evaluates "
        "the performance of different approaches to determine "
        "the most profitable and risk-averse options."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    cache=False,
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies "
    "based on approved trading strategies.",
    backstory=(
        "This agent specializes in analyzing the timing, price, "
        "and logistical details of potential trades. By evaluating "
        "these factors, it provides well-founded suggestions for "
        "when and how trades should be executed to maximize "
        "efficiency and adherence to strategy."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    cache=False,
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
    "associated with potential trading activities.",
    backstory=(
        "Armed with a deep understanding of risk assessment models "
        "and market dynamics, this agent scrutinizes the potential "
        "risks of proposed trades. It offers a detailed analysis of "
        "risk exposure and suggests safeguards to ensure that "
        "trading activities align with the firm's risk tolerance."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    cache=False,
)


# =============================================================================
# Tasks
# =============================================================================

data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the "
        "best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)


# =============================================================================
# Crew
# =============================================================================

financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent,
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task,
    ],
    process=Process.sequential,
    verbose=False,
    memory=False,  # Disable memory to avoid OpenAI embedding calls
)

# Input parameters for the crew
financial_trading_inputs = {
    "stock_selection": "CSCO",
    "initial_capital": "100000",
    "risk_tolerance": "Medium",
    "trading_strategy_preference": "Day Trading",
    "news_impact_consideration": True,
}


# =============================================================================
# Telemetry Flush (for manual mode)
# =============================================================================

def flush_telemetry():
    """Flush all OpenTelemetry providers before exit."""
    if not MANUAL_INSTRUMENTATION:
        return  # Zero-code mode handles this via run.sh

    from opentelemetry import trace, metrics

    print("\n[FLUSH] Starting telemetry flush", flush=True)

    # Flush traces
    try:
        tp = trace.get_tracer_provider()
        if hasattr(tp, "force_flush"):
            print("[FLUSH] Flushing traces (timeout=30s)", flush=True)
            tp.force_flush(timeout_millis=30000)
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush traces: {e}", flush=True)

    # Flush metrics
    try:
        mp = metrics.get_meter_provider()
        if hasattr(mp, "force_flush"):
            print("[FLUSH] Flushing metrics (timeout=30s)", flush=True)
            mp.force_flush(timeout_millis=30000)
        if hasattr(mp, "shutdown"):
            print("[FLUSH] Shutting down metrics provider", flush=True)
            mp.shutdown()
    except Exception as e:
        print(f"[FLUSH] Warning: Could not flush metrics: {e}", flush=True)

    time.sleep(2)
    print("[FLUSH] Telemetry flush complete\n", flush=True)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    exit_code = 0
    try:
        # Refresh LLM with fresh token if using OAuth2
        if USE_OAUTH2:
            fresh_llm = get_llm()
            # Update all agents with fresh LLM
            data_analyst_agent.llm = fresh_llm
            trading_strategy_agent.llm = fresh_llm
            execution_agent.llm = fresh_llm
            risk_management_agent.llm = fresh_llm
            print("[AUTH] OAuth2 token refreshed")

        print(f"\n[START] Executing Financial Trading Crew")
        print(f"[START] Stock: {financial_trading_inputs['stock_selection']}")
        print(f"[START] Risk Tolerance: {financial_trading_inputs['risk_tolerance']}")
        print(f"[START] Strategy: {financial_trading_inputs['trading_strategy_preference']}")
        print("")

        result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

        print("\n[SUCCESS] Crew execution completed")
        print("=" * 80)
        print(result)

    except Exception as e:
        print(f"\n[ERROR] Crew execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        if MANUAL_INSTRUMENTATION:
            print("\n" + "=" * 80)
            print("TELEMETRY OUTPUT - Look for gen_ai.workflow and gen_ai.agent spans")
            print("=" * 80 + "\n")
        flush_telemetry()
        sys.exit(exit_code)


# =============================================================================
# Expected Trace Structure (Sequential Process):
# =============================================================================
# gen_ai.workflow (Financial Trading Crew)
# ├── gen_ai.step (Data Analysis)
# │   └── invoke_agent (Data Analyst)
# │       ├── chat (OpenAI/LiteLLM)
# │       │   └── gen_ai.choice
# │       ├── tool (Search the internet)
# │       └── tool (Read website content)
# ├── gen_ai.step (Strategy Development)
# │   └── invoke_agent (Trading Strategy Developer)
# │       ├── chat (OpenAI/LiteLLM)
# │       └── tool (Search the internet)
# ├── gen_ai.step (Execution Planning)
# │   └── invoke_agent (Trade Advisor)
# │       └── chat (OpenAI/LiteLLM)
# └── gen_ai.step (Risk Assessment)
#     └── invoke_agent (Risk Advisor)
#         ├── chat (OpenAI/LiteLLM)
#         └── tool (Read website content)
# =============================================================================

