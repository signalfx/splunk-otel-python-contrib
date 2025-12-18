# Key Features:
# SRE Agent that analyzes alerts using all available data sources
#
#
# Three main endpoints:
# - /api/sre/analyze - Analyzes specific alerts with trace correlation
# - /api/sre/investigate - Investigates services for abnormalities
# - /api/sre/query - General purpose queries
#
#
# Intelligent Analysis:
# - Correlates alerts with traces via trace_id
# - Retrieves runbook guidance automatically
# - Analyzes logs and metrics for root cause
# - Provides actionable remediation steps
#
#
# Vector Search across all data types for semantic retrieval
# # Analyze a specific alert
# curl -X POST http://localhost:5000/api/sre/analyze \
#   -H "Content-Type: application/json" \
#   -d '{"alert_id": "search_timeout_critical"}'
#
# # Investigate a service
# curl -X POST http://localhost:5000/api/sre/investigate \
#   -H "Content-Type: application/json" \
#   -d '{"service": "checkout-service"}'
#
# # General query
# curl -X POST http://localhost:5000/api/sre/query \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What are the critical issues across all services?"}'

import os
import json
from typing import Annotated, TypedDict, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
import operator
import weaviate
from flask import Flask, request, jsonify
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)


# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    retrieved_docs: List[Document]


# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global data stores
alerts_data = []
traces_data = []
metrics_data = []
logs_data = []
runbook_data = []

# Weaviate client
weaviate_client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_PORT,
    http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
    grpc_host=WEAVIATE_GRPC_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
)

embeddings_model = OpenAIEmbeddings()

# Vector stores for different data types
alerts_vectorstore = None
runbook_vectorstore = None
traces_vectorstore = None
logs_vectorstore = None
metrics_vectorstore = None


# Load data functions
def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def load_alerts_data():
    """Load alerts into vector store."""
    global alerts_data, alerts_vectorstore

    alerts_data = load_json_file("alerts.json")
    documents = []

    for alert in alerts_data:
        alert_details = alert.get("alert", {})
        content = f"""
Alert ID: {alert.get('id')}
Title: {alert.get('title')}
Service: {alert_details.get('service')}
Severity: {alert_details.get('severity')}
Description: {alert_details.get('description')}
Trace ID: {alert_details.get('trace_id')}
        """.strip()

        documents.append(Document(
            page_content=content,
            metadata={**alert, "type": "alert"}
        ))

    alerts_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings_model,
        index_name="Alerts",
        text_key="content"
    )
    alerts_vectorstore.add_documents(documents)
    print(f"Loaded {len(documents)} alerts")


def load_runbook_data():
    """Load runbooks into vector store."""
    global runbook_data, runbook_vectorstore

    runbook_data = load_json_file("runbook.json")
    documents = []

    for runbook in runbook_data:
        for section in runbook.get("sections", []):
            content = f"""
Runbook ID: {runbook.get('id')}
Title: {runbook.get('title')}
Scenario: {runbook.get('scenario')}
Services: {', '.join(runbook.get('services', []))}
Severity: {', '.join(runbook.get('severity', []))}
Section: {section.get('title')}
Content: {section.get('content')}
            """.strip()

            documents.append(Document(
                page_content=content,
                metadata={
                    **runbook,
                    "section_id": section.get('section_id'),
                    "type": "runbook"
                }
            ))

    runbook_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings_model,
        index_name="Runbooks",
        text_key="content"
    )
    runbook_vectorstore.add_documents(documents)
    print(f"Loaded {len(documents)} runbook sections")


def load_traces_data():
    """Load traces into vector store."""
    global traces_data, traces_vectorstore

    traces_data = load_json_file("otel-traces.json")
    documents = []

    for trace in traces_data:
        content = f"""
Trace ID: {trace.get('trace_id')}
Span: {trace.get('name')}
Service: {trace.get('resource', {}).get('service.name')}
Duration: {trace.get('duration_ms')}ms
Status: {trace.get('status', {}).get('code')}
Attributes: {json.dumps(trace.get('attributes', {}))}
        """.strip()

        documents.append(Document(
            page_content=content,
            metadata={**trace, "type": "trace"}
        ))

    traces_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings_model,
        index_name="Traces",
        text_key="content"
    )
    traces_vectorstore.add_documents(documents)
    print(f"Loaded {len(documents)} traces")


def load_logs_data():
    """Load logs into vector store."""
    global logs_data, logs_vectorstore

    logs_data = load_json_file("otel-logs.json")
    documents = []

    for log in logs_data:
        content = f"""
Timestamp: {log.get('timestamp')}
Service: {log.get('resource', {}).get('service.name')}
Severity: {log.get('severity_text')}
Message: {log.get('body')}
Trace ID: {log.get('trace_id')}
Attributes: {json.dumps(log.get('attributes', {}))}
        """.strip()

        documents.append(Document(
            page_content=content,
            metadata={**log, "type": "log"}
        ))

    logs_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings_model,
        index_name="Logs",
        text_key="content"
    )
    logs_vectorstore.add_documents(documents)
    print(f"Loaded {len(documents)} logs")


def load_metrics_data():
    """Load metrics into vector store."""
    global metrics_data, metrics_vectorstore

    metrics_data = load_json_file("otel-metrics.json")
    documents = []

    for metric in metrics_data:
        for dp in metric.get('data_points', []):
            content = f"""
Metric: {metric.get('metric_name')}
Service: {metric.get('resource', {}).get('service.name')}
Value: {dp.get('value')}{dp.get('unit', '')}
Timestamp: {dp.get('timestamp')}
Attributes: {json.dumps(dp.get('attributes', {}))}
            """.strip()

            documents.append(Document(
                page_content=content,
                metadata={**metric, **dp, "type": "metric"}
            ))

    metrics_vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings_model,
        index_name="Metrics",
        text_key="content"
    )
    metrics_vectorstore.add_documents(documents)
    print(f"Loaded {len(documents)} metric data points")


# MCP Tools
@tool
def retrieve_alerts(query: str, k: int = 3) -> str:
    """Retrieve relevant alerts from the knowledge base."""
    if not alerts_vectorstore:
        return "Error: Alerts not loaded"

    results = alerts_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([f"Alert {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


@tool
def retrieve_runbook(query: str, k: int = 3, service_filter: str = None) -> str:
    """Retrieve runbook guidance for troubleshooting."""
    if not runbook_vectorstore:
        return "Error: Runbooks not loaded"

    results = runbook_vectorstore.similarity_search(query, k=k)

    if service_filter:
        results = [doc for doc in results if service_filter in doc.metadata.get('services', [])]

    return "\n\n".join([f"Runbook {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


@tool
def retrieve_traces(query: str, k: int = 3) -> str:
    """Retrieve trace data for analysis."""
    if not traces_vectorstore:
        return "Error: Traces not loaded"

    results = traces_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([f"Trace {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


@tool
def retrieve_logs(query: str, k: int = 3) -> str:
    """Retrieve log entries for investigation."""
    if not logs_vectorstore:
        return "Error: Logs not loaded"

    results = logs_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([f"Log {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


@tool
def retrieve_metrics(query: str, k: int = 3) -> str:
    """Retrieve metrics data for performance analysis."""
    if not metrics_vectorstore:
        return "Error: Metrics not loaded"

    results = metrics_vectorstore.similarity_search(query, k=k)
    return "\n\n".join([f"Metric {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)])


# LangGraph setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([
    retrieve_alerts,
    retrieve_runbook,
    retrieve_traces,
    retrieve_logs,
    retrieve_metrics
])


def invoke_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", invoke_model)
workflow.add_node("tools", ToolNode([
    retrieve_alerts,
    retrieve_runbook,
    retrieve_traces,
    retrieve_logs,
    retrieve_metrics
]))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

sre_agent = workflow.compile()


# Flask API endpoints
@app.route('/api/sre/analyze', methods=['POST'])
def analyze_alert():
    """SRE agent analyzes an alert and provides resolution."""
    data = request.json
    alert_id = data.get('alert_id')

    if not alert_id:
        return jsonify({"error": "alert_id required"}), 400

    # Find the alert
    alert = next((a for a in alerts_data if a.get('id') == alert_id), None)
    if not alert:
        return jsonify({"error": "Alert not found"}), 404

    # Create analysis query
    query = f"""
Analyze this alert and provide a comprehensive resolution:

Alert: {alert.get('title')}
Service: {alert.get('alert', {}).get('service')}
Severity: {alert.get('alert', {}).get('severity')}
Description: {alert.get('alert', {}).get('description')}

Please:
1. Retrieve relevant runbook guidance
2. Check traces for the trace_id: {alert.get('alert', {}).get('trace_id')}
3. Review related logs and metrics
4. Identify root cause and abnormalities
5. Suggest specific remediation steps
    """

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "retrieved_docs": []
    }

    result = sre_agent.invoke(initial_state)
    response = result["messages"][-1].content

    return jsonify({
        "alert_id": alert_id,
        "analysis": response,
        "alert_details": alert
    })


@app.route('/api/sre/investigate', methods=['POST'])
def investigate_service():
    """Investigate a service for abnormalities."""
    data = request.json
    service_name = data.get('service')

    if not service_name:
        return jsonify({"error": "service name required"}), 400

    query = f"""
Investigate {service_name} for any abnormalities or issues:

1. Check all alerts related to {service_name}
2. Review recent traces for errors or high latency
3. Analyze logs for errors or warnings
4. Check metrics for performance degradation
5. Provide a summary of findings and recommendations
    """

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "retrieved_docs": []
    }

    result = sre_agent.invoke(initial_state)
    response = result["messages"][-1].content

    return jsonify({
        "service": service_name,
        "investigation": response
    })


@app.route('/api/sre/query', methods=['POST'])
def query_sre():
    """General SRE query endpoint."""
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "query required"}), 400

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "retrieved_docs": []
    }

    result = sre_agent.invoke(initial_state)
    response = result["messages"][-1].content

    return jsonify({"result": response})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "alerts_loaded": len(alerts_data),
        "runbooks_loaded": len(runbook_data),
        "traces_loaded": len(traces_data),
        "logs_loaded": len(logs_data),
        "metrics_loaded": len(metrics_data)
    })


def initialize():
    """Initialize all data stores."""
    print("Initializing SRE Agent...")
    load_alerts_data()
    load_runbook_data()
    load_traces_data()
    load_logs_data()
    load_metrics_data()
    print("Initialization complete!")


if __name__ == "__main__":
    try:
        initialize()
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        weaviate_client.close()