import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()


# Set environment variables for LangChain/Langsmith native OTEL export
# This is the ONLY instrumentation source - no LangChainInstrumentor needed
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


# Configure the OTLP exporter for your custom endpoint
provider = TracerProvider()

# Then add the OTLP exporter to send the translated spans
otlp_exporter = OTLPSpanExporter(
    # Change to your provider's endpoint
    endpoint="http://localhost:4318/v1/traces",
)
batch_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(batch_processor)
trace.set_tracer_provider(provider)

# NOTE: We intentionally do NOT use LangChainInstrumentor here.
# Native Langsmith OTEL export (LANGSMITH_OTEL_ENABLED=true) creates spans,
# and our LangsmithSpanProcessor translates their attributes to GenAI semconv.
# Using both would create duplicate spans!


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com",
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com",
    },
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"


model = ChatOpenAI(model="gpt-4o", temperature=0.7)
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123"),
)
print(result)
