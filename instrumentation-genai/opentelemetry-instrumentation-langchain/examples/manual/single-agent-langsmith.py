import os
import random
from typing import TypedDict, Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langsmith import trace

# Configure tracing/metrics/logging once per process so exported data goes to OTLP.
# from opentelemetry import _events, _logs, metrics, trace
# from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
# from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
#     OTLPMetricExporter,
# )
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
#     OTLPSpanExporter,
# )
# from opentelemetry.instrumentation.langchain import LangchainInstrumentor
# from opentelemetry.sdk._events import EventLoggerProvider
# from opentelemetry.sdk._logs import LoggerProvider
# from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
# from opentelemetry.sdk.metrics import MeterProvider
# from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor


# trace.set_tracer_provider(TracerProvider())
# trace.get_tracer_provider().add_span_processor(
#     BatchSpanProcessor(OTLPSpanExporter())
# )

# demo_tracer = trace.get_tracer("instrumentation.langchain.demo")

# metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
# metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# _logs.set_logger_provider(LoggerProvider())
# _logs.get_logger_provider().add_log_record_processor(
#     BatchLogRecordProcessor(OTLPLogExporter())
# )
# _events.set_event_logger_provider(EventLoggerProvider())


#---- define the agent graph nodes and tools ----#

# Define the state
class AgentState(TypedDict):
    input: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    output: str

# Define the capital lookup tool
@tool
def get_capital(country: str) -> str:
    """Return the capital city for the given country name."""
    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
        "japan": "Tokyo",
        "canada": "Ottawa",
        "australia": "Canberra",
        "brazil": "Brasília",
        "india": "New Delhi",
        "united states": "Washington, D.C.",
        "united kingdom": "London",
    }
    return capitals.get(country.strip().lower(), "Unknown")

# Capital subagent node
def capital_subagent(state: AgentState) -> AgentState:
    question = state["input"]
    # Simple extraction (last word); in production, use parsing or LLM
    country = question.rstrip("?!. ").split(" ")[-1]
    cap = get_capital(country)
    if cap == "Unknown":
        answer = f"Sorry, I don't know the capital of {country.capitalize()}."
    else:
        answer = f"The capital of {country.capitalize()} is {cap}."
    return {"messages": [AIMessage(content=answer)], "output": answer}

# General LLM node
def general_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    question = state["input"]
    system_prompt = "You are a helpful, concise assistant."
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])
    return {"messages": [response], "output": response.content}

# Classifier node (deterministic routing)
def classifier(state: AgentState) -> AgentState:
    q = state["input"].lower()
    route = "capital" if ("capital" in q or "city" in q) else "general"
    return {"route": route}

# Route decider for conditional edges
def route_decider(state: AgentState) -> str:
    return state.get("route", "general")

#---- build and compile the agent graph ----#

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("classify", classifier)
graph.add_node("capital_agent", capital_subagent)
graph.add_node("general_agent", general_node)

graph.set_entry_point("classify")
graph.add_conditional_edges(
    "classify",
    route_decider,
    {"capital": "capital_agent", "general": "general_agent"},
)
graph.add_edge("capital_agent", END)
graph.add_edge("general_agent", END)

# Compile the app
app = graph.compile()

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key as env var: export OPENAI_API_KEY="sk-..."
    queries = [
        "What is the capital of France?",
        "Explain why the sky is blue in one sentence.",
        "What is the capital city of Brazil?",
    ]
    # instrumentor = LangchainInstrumentor()
    # instrumentor.instrument()

    # tracer = trace.get_tracer("demo.manual.langchain.single")
    # with tracer.start_as_current_span("run-queries", kind=trace.SpanKind.SERVER):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4318/v1/traces"
    )

    provider = TracerProvider()
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    
    query = random.choice(queries)
    result = app.invoke({"input": query, "messages": []}) # type: ignore
    print(f"Query: {query}")
    print(f"Output: {result['output']}\n")
    
    # instrumentor.uninstrument()