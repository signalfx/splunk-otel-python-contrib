"""
Retail Shop Chain Application - LangChain Automatic Instrumentation
====================================================================

Architecture:
  Store Manager (Parent)
  ├─ Inventory Agent (Child 1)
  └─ Customer Service Agent (Child 2)

This app uses LangChain's automatic instrumentation with create_agent()
and LangchainInstrumentor().instrument() to demonstrate evaluation metrics.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import logging
import time

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ============================================================================
# OpenTelemetry Setup
# ============================================================================
# Configure resource - DO NOT set service.name or deployment.environment here
# They are automatically picked up from OTEL_SERVICE_NAME and OTEL_DEPLOYMENT_ENVIRONMENT
resource = Resource.create({
    "agent.name": "retail-chain",
    "agent.type": "multi-agent-retail",
})

# Configure tracing
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# Configure metrics
metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader], resource=resource))

# Configure logging
logger_provider = LoggerProvider(resource=resource)
_logs.set_logger_provider(logger_provider)
log_processor = BatchLogRecordProcessor(OTLPLogExporter())
logger_provider.add_log_record_processor(log_processor)
handler = LoggingHandler(level=logging.WARNING, logger_provider=logger_provider)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)

# Configure events
_events.set_event_logger_provider(EventLoggerProvider())

# ============================================================================
# Instrument LangChain (AUTOMATIC INSTRUMENTATION)
# ============================================================================
instrumentor = LangchainInstrumentor()
instrumentor.instrument()

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Functions for Agents
# ============================================================================
def check_inventory(product_name: str) -> str:
    """Check if a product is in stock."""
    # Mock inventory check
    inventory = {
        "iphone 15 pro": "In stock - 15 units available in Space Black, Natural Titanium, Blue Titanium",
        "macbook pro": "In stock - 8 units available in Space Gray and Silver",
        "airpods pro": "In stock - 25 units available",
        "laptop": "In stock - 12 units available across multiple brands",
    }
    
    for key in inventory:
        if key in product_name.lower():
            return inventory[key]
    
    return f"Product '{product_name}' - Please check with store manager for availability"


def get_return_policy(product_type: str) -> str:
    """Get return policy for a product type."""
    policies = {
        "electronics": "30-day return policy. Product must be unopened with original packaging. Restocking fee may apply.",
        "laptop": "14-day return policy for laptops. Must include all accessories and original packaging.",
        "phone": "14-day return policy. Device must be in original condition with no signs of use.",
    }
    
    for key in policies:
        if key in product_type.lower():
            return policies[key]
    
    return "Standard 30-day return policy applies. Please bring receipt and product in original condition."


def format_response(text: str) -> str:
    """Format response for customer."""
    return f"**Customer Response:**\n{text}"


# ============================================================================
# Retail Shop Application
# ============================================================================
def run_retail_scenario(scenario_name: str, customer_request: str, llm: ChatOpenAI):
    """Run a single retail shop scenario with parent and child agents."""
    
    print("\n" + "=" * 80)
    print(f"🏪 {scenario_name}")
    print("=" * 80)
    print(f"Customer Request: {customer_request}")
    print()
    
    # ========================================================================
    # Create Child Agent 1: Inventory Agent
    # ========================================================================
    inventory_agent = create_agent(
        name="inventory-agent",
        model=llm,
        tools=[check_inventory],
        system_prompt="You are an inventory specialist. Check stock levels and provide accurate availability information.",
        debug=False,
    ).with_config({
        "run_name": "inventory-agent",
        "tags": ["agent:inventory", "agent", "order:1"],
        "metadata": {
            "agent_name": "inventory-agent",
            "agent_role": "inventory_specialist",
            "agent_order": 1,
        }
    })
    
    # ========================================================================
    # Create Child Agent 2: Customer Service Agent
    # ========================================================================
    customer_service_agent = create_agent(
        name="customer-service-agent",
        model=llm,
        tools=[get_return_policy, format_response],
        system_prompt="You are a friendly customer service representative. Help customers with inquiries professionally and courteously.",
        debug=False,
    ).with_config({
        "run_name": "customer-service-agent",
        "tags": ["agent:customer-service", "agent", "order:2"],
        "metadata": {
            "agent_name": "customer-service-agent",
            "agent_role": "customer_service_rep",
            "agent_order": 2,
        }
    })
    
    # ========================================================================
    # Create Parent Agent: Store Manager
    # ========================================================================
    store_manager_agent = create_agent(
        name="store-manager",
        model=llm,
        tools=[],  # Manager coordinates but doesn't use tools directly
        system_prompt="You are a store manager. Coordinate with inventory and customer service teams to help customers. Synthesize information from both teams.",
        debug=False,
    ).with_config({
        "run_name": "store-manager",
        "tags": ["agent:manager", "agent", "order:0"],
        "metadata": {
            "agent_name": "store-manager",
            "agent_role": "store_coordinator",
            "agent_order": 0,
        }
    })
    
    # ========================================================================
    # Execute Workflow: Parent → Child 1 → Child 2
    # ========================================================================
    try:
        # Step 1: Inventory Agent checks stock
        print("⏳ Inventory Agent checking stock...", end="", flush=True)
        inventory_result = inventory_agent.invoke(
            {"messages": [{"role": "user", "content": customer_request}]},
            {"session_id": f"{scenario_name}-inventory"}
        )
        
        if inventory_result and "messages" in inventory_result:
            final_message = inventory_result["messages"][-1]
            inventory_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        else:
            inventory_response = str(inventory_result)
        
        print(f" ✓ ({len(inventory_response)} chars)")
        
        # Step 2: Customer Service Agent handles the request
        print("⏳ Customer Service Agent responding...", end="", flush=True)
        service_prompt = f"""Customer Request: {customer_request}

Inventory Information: {inventory_response}

Please provide a helpful customer service response."""
        
        service_result = customer_service_agent.invoke(
            {"messages": [{"role": "user", "content": service_prompt}]},
            {"session_id": f"{scenario_name}-service"}
        )
        
        if service_result and "messages" in service_result:
            final_message = service_result["messages"][-1]
            service_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        else:
            service_response = str(service_result)
        
        print(f" ✓ ({len(service_response)} chars)")
        
        # Step 3: Store Manager synthesizes
        print("⏳ Store Manager synthesizing...", end="", flush=True)
        manager_prompt = f"""Customer Request: {customer_request}

Inventory Team Response: {inventory_response}

Customer Service Team Response: {service_response}

As store manager, provide a final coordinated response to the customer."""
        
        manager_result = store_manager_agent.invoke(
            {"messages": [{"role": "user", "content": manager_prompt}]},
            {"session_id": f"{scenario_name}-manager"}
        )
        
        if manager_result and "messages" in manager_result:
            final_message = manager_result["messages"][-1]
            manager_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        else:
            manager_response = str(manager_result)
        
        print(f" ✓ ({len(manager_response)} chars)")
        
        # Display final response
        print("\n" + "-" * 80)
        print("📝 Store Manager Final Response:")
        print("-" * 80)
        print(manager_response[:500] + ("..." if len(manager_response) > 500 else ""))
        print("-" * 80)
        
        print(f"\n✅ {scenario_name} Complete")
        
    except Exception:
        print(f"\n❌ Error in {scenario_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run retail shop scenarios with LangChain automatic instrumentation."""
    print("=" * 80)
    print("🏪 RETAIL SHOP CHAIN - LANGCHAIN AUTOMATIC INSTRUMENTATION")
    print("=" * 80)
    print("Architecture: Store Manager → Inventory Agent + Customer Service Agent")
    print("Instrumentation: LangchainInstrumentor().instrument()")
    print("=" * 80)
    print()
    print("🔧 Configuration:")
    print("  Service: retail-shop-langchain")
    print("  Environment: alpha-test")
    print(f"  Deepeval API Key: {'SET' if os.getenv('DEEPEVAL_API_KEY') else 'NOT SET'}")
    print("=" * 80)
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        temperature=0.7,
        max_tokens=500
    )
    
    # ========================================================================
    # SCENARIO 1: Product Availability Inquiry (UNIFIED TRACE)
    # ========================================================================
    tracer = trace.get_tracer(__name__)
    
    print("\n🔵 Starting Scenario 1 with unified trace...")
    with tracer.start_as_current_span("retail_workflow_scenario_1") as root_span:
        root_span.set_attribute("scenario.name", "Product Availability")
        root_span.set_attribute("scenario.type", "product_inquiry")
        root_span.set_attribute("workflow.type", "retail_shop")
        
        # Get trace ID for reporting
        trace_id = format(root_span.get_span_context().trace_id, '032x')
        print(f"   Trace ID: {trace_id}")
        
        run_retail_scenario(
            scenario_name="Scenario 1: Product Availability",
            customer_request="Do you have the new iPhone 15 Pro in stock? What colors are available?",
            llm=llm
        )
    
    print(f"✅ Scenario 1 Complete - Trace ID: {trace_id}")
    
    # ========================================================================
    # SCENARIO 2: Return Request (UNIFIED TRACE)
    # ========================================================================
    print("\n🔵 Starting Scenario 2 with unified trace...")
    with tracer.start_as_current_span("retail_workflow_scenario_2") as root_span:
        root_span.set_attribute("scenario.name", "Product Return")
        root_span.set_attribute("scenario.type", "return_request")
        root_span.set_attribute("workflow.type", "retail_shop")
        
        # Get trace ID for reporting
        trace_id = format(root_span.get_span_context().trace_id, '032x')
        print(f"   Trace ID: {trace_id}")
        
        run_retail_scenario(
            scenario_name="Scenario 2: Product Return",
            customer_request="I need to return a laptop I purchased last week. What's the process?",
            llm=llm
        )
    
    print(f"✅ Scenario 2 Complete - Trace ID: {trace_id}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("✅ ALL SCENARIOS COMPLETE")
    print("=" * 80)
    print("Total Scenarios: 2")
    print("Architecture: 1 Parent + 2 Children = 3 Agents per scenario")
    print("Instrumentation: LangChain Automatic (LangchainInstrumentor)")
    print()
    print("Expected Results:")
    print("  ✅ 2 traces (one per scenario)")
    print("  ✅ Each trace shows: Store Manager → Inventory + Customer Service")
    print("  ✅ Evaluation metrics on ALL 3 agents")
    print("  ✅ All metrics should PASS (normal content)")
    print()
    print("Validation:")
    print("  [ ] Both traces visible in Splunk APM")
    print("  [ ] Each trace shows 3 agent invocations")
    print("  [ ] Evaluation metrics visible on all agents")
    print("  [ ] Service name: retail-shop-langchain")
    print("=" * 80)
    
    # Wait for evaluations (matching langgraph app)
    print("\n⏳ Waiting 300 seconds for telemetry export and async evaluations...")
    print("   (LangChain automatic instrumentation + Deepeval evaluations)")
    
    # Flush telemetry
    print("\n📤 Flushing telemetry providers...")
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    
    # Wait for async evaluations
    time.sleep(300)
    
    # Final flush
    print("\n📤 Final flush...")
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
