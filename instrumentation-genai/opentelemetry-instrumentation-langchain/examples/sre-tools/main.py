# Flask REST APIs for querying alerts, traces, metrics, and logs
# Separate vector stores for each data type (alerts, traces, metrics, logs)
# Tools (retrieve_alerts, retrieve_traces, retrieve_metrics, retrieve_logs)
# Correlation via trace IDs between different telemetry signals
# Natural language queries powered by LangGraph and GPT-4
# API endpoints:
# POST /api/query/alerts - Query alerts
# POST /api/query/traces - Query traces
# POST /api/query/metrics - Query metrics
# POST /api/query/logs - Query logs
# POST /api/query - Query all data sources
# GET /health - Health check

import os
import json
from typing import List
from pathlib import Path
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
from typing import Annotated, TypedDict, Sequence
import operator
from langchain_core.messages import BaseMessage
import weaviate

# Flask app
app = Flask(__name__)

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
alerts_store = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings_model,
    index_name="Alerts",
    text_key="content"
)

traces_store = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings_model,
    index_name="Traces",
    text_key="content"
)

metrics_store = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings_model,
    index_name="Metrics",
    text_key="content"
)

logs_store = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings_model,
    index_name="Logs",
    text_key="content"
)

runbook_store = WeaviateVectorStore(
    client=weaviate_client,
    embedding=embeddings_model,
    index_name="Runbook",
    text_key="content"
)

# Agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    retrieved_docs: List[Document]


def load_alerts_data(json_file_path: str = "./alerts.json") -> None:
    """Load alerts JSON into vector store."""
    print(f"Loading alerts from {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    documents = []
    for alert_obj in data:
        alert_id = alert_obj.get("id", "unknown")
        title = alert_obj.get("title", "")
        alert_details = alert_obj.get("alert", {})

        content = f"""
Alert ID: {alert_id}
Title: {title}
Service: {alert_details.get('service', 'N/A')}
Environment: {alert_details.get('environment', 'N/A')}
Region: {alert_details.get('region', 'N/A')}
Severity: {alert_details.get('severity', 'N/A')}
Alert Name: {alert_details.get('name', 'N/A')}
Window: {alert_details.get('window_minutes', 'N/A')} minutes
Description: {alert_details.get('description', 'N/A')}
Trace ID: {alert_details.get('trace_id', 'N/A')}
        """.strip()

        metadata = {
            "alert_id": alert_id,
            "title": title,
            "service": alert_details.get('service', ''),
            "environment": alert_details.get('environment', ''),
            "severity": alert_details.get('severity', ''),
            "trace_id": alert_details.get('trace_id', ''),
            "type": "alert"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    alerts_store.add_documents(documents)
    print(f"Loaded {len(documents)} alerts")


def load_traces_data(json_file_path: str = "./otel-traces.json") -> None:
    """Load traces JSON into vector store."""
    print(f"Loading traces from {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    documents = []
    for trace in data:
        trace_id = trace.get("trace_id", "unknown")
        span_id = trace.get("span_id", "unknown")
        resource = trace.get("resource", {})
        attributes = trace.get("attributes", {})

        content = f"""
Trace ID: {trace_id}
Span ID: {span_id}
Name: {trace.get('name', 'N/A')}
Kind: {trace.get('kind', 'N/A')}
Service: {resource.get('service.name', 'N/A')}
Environment: {resource.get('deployment.environment', 'N/A')}
Duration: {trace.get('duration_ms', 'N/A')}ms
Status: {trace.get('status', {}).get('code', 'N/A')}
HTTP Method: {attributes.get('http.method', 'N/A')}
HTTP Status: {attributes.get('http.status_code', 'N/A')}
Error: {attributes.get('error', False)}
        """.strip()

        metadata = {
            "trace_id": trace_id,
            "span_id": span_id,
            "service": resource.get('service.name', ''),
            "environment": resource.get('deployment.environment', ''),
            "duration_ms": trace.get('duration_ms', 0),
            "status": trace.get('status', {}).get('code', ''),
            "error": attributes.get('error', False),
            "type": "trace"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    traces_store.add_documents(documents)
    print(f"Loaded {len(documents)} traces")


def load_metrics_data(json_file_path: str = "./otel-metrics.json") -> None:
    """Load metrics JSON into vector store."""
    print(f"Loading metrics from {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    documents = []
    for metric in data:
        metric_id = metric.get("id", "unknown")
        metric_name = metric.get("metric_name", "unknown")
        resource = metric.get("resource", {})
        data_points = metric.get("data_points", [])

        for dp in data_points:
            content = f"""
Metric ID: {metric_id}
Metric Name: {metric_name}
Service: {resource.get('service.name', 'N/A')}
Environment: {resource.get('deployment.environment', 'N/A')}
Timestamp: {dp.get('timestamp', 'N/A')}
Value: {dp.get('value', 'N/A')} {dp.get('unit', '')}
Trace ID: {dp.get('attributes', {}).get('trace_id', 'N/A')}
            """.strip()

            metadata = {
                "metric_id": metric_id,
                "metric_name": metric_name,
                "service": resource.get('service.name', ''),
                "environment": resource.get('deployment.environment', ''),
                "value": dp.get('value', 0),
                "trace_id": dp.get('attributes', {}).get('trace_id', ''),
                "type": "metric"
            }

            documents.append(Document(page_content=content, metadata=metadata))

    metrics_store.add_documents(documents)
    print(f"Loaded {len(documents)} metric data points")


def load_logs_data(json_file_path: str = "./otel-logs.json") -> None:
    """Load logs JSON into vector store."""
    print(f"Loading logs from {json_file_path}")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    documents = []
    for log in data:
        trace_id = log.get("trace_id", "unknown")
        span_id = log.get("span_id", "unknown")
        resource = log.get("resource", {})
        attributes = log.get("attributes", {})

        content = f"""
Trace ID: {trace_id}
Span ID: {span_id}
Timestamp: {log.get('timestamp', 'N/A')}
Severity: {log.get('severity_text', 'N/A')}
Service: {resource.get('service.name', 'N/A')}
Environment: {resource.get('deployment.environment', 'N/A')}
Body: {log.get('body', 'N/A')}
Log Source: {attributes.get('log.source', 'N/A')}
        """.strip()

        metadata = {
            "trace_id": trace_id,
            "span_id": span_id,
            "service": resource.get('service.name', ''),
            "environment": resource.get('deployment.environment', ''),
            "severity": log.get('severity_text', ''),
            "log_source": attributes.get('log.source', ''),
            "type": "log"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    logs_store.add_documents(documents)
    print(f"Loaded {len(documents)} logs")

def load_runbook_data(json_file_path: str = "./runbook.json") -> None:
    """Load runbook JSON into vector store."""
    try:
        file_path = Path(json_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Runbook file not found at {json_file_path}")

        with open(file_path, "r") as f:
            runbooks = json.load(f)

        documents = []
        for rb in runbooks:
            rb_id = rb.get("id", "unknown")
            title = rb.get("title", "")
            scenario = rb.get("scenario", "")
            services = ", ".join(rb.get("services", []))
            severity = ", ".join(rb.get("severity", []))

            # Create documents for each section
            for section in rb.get("sections", []):
                section_id = section.get("section_id", "")
                section_title = section.get("title", "")
                content = section.get("content", "")

                page_content = f"""
Runbook ID: {rb_id}
Title: {title}
Scenario: {scenario}
Services: {services}
Severity Levels: {severity}
Section: {section_title}
Content: {content}
                """.strip()

                metadata = {
                    "runbook_id": rb_id,
                    "section_id": section_id,
                    "title": title,
                    "scenario": scenario,
                    "section_title": section_title,
                    "services": services,
                    "severity": severity
                }

                documents.append(Document(page_content=page_content, metadata=metadata))

        runbook_store.add_documents(documents)
        print(f"Indexed {len(documents)} runbook sections into vector store")

    except Exception as e:
        print(f"Error loading runbook data: {e}")
        raise

# Tools
@tool
def retrieve_alerts(query: str, k: int = 3, severity_filter: str = None) -> str:
    """Retrieve relevant alerts from the knowledge base."""
    try:
        results = alerts_store.similarity_search(query, k=k)

        if severity_filter:
            results = [doc for doc in results if doc.metadata.get('severity', '').lower() == severity_filter.lower()]

        if not results:
            return "No relevant alerts found."

        context = "\n\n".join([f"[Alert {i + 1}]\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Retrieved {len(results)} alert(s):\n\n{context}"
    except Exception as e:
        return f"Error retrieving alerts: {str(e)}"


@tool
def retrieve_traces(query: str, k: int = 3, trace_id: str = None) -> str:
    """Retrieve relevant traces from the knowledge base."""
    try:
        results = traces_store.similarity_search(query, k=k)

        if trace_id:
            results = [doc for doc in results if doc.metadata.get('trace_id') == trace_id]

        if not results:
            return "No relevant traces found."

        context = "\n\n".join([f"[Trace {i + 1}]\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Retrieved {len(results)} trace(s):\n\n{context}"
    except Exception as e:
        return f"Error retrieving traces: {str(e)}"


@tool
def retrieve_metrics(query: str, k: int = 3, metric_name: str = None) -> str:
    """Retrieve relevant metrics from the knowledge base."""
    try:
        results = metrics_store.similarity_search(query, k=k)

        if metric_name:
            results = [doc for doc in results if doc.metadata.get('metric_name') == metric_name]

        if not results:
            return "No relevant metrics found."

        context = "\n\n".join([f"[Metric {i + 1}]\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Retrieved {len(results)} metric(s):\n\n{context}"
    except Exception as e:
        return f"Error retrieving metrics: {str(e)}"


@tool
def retrieve_logs(query: str, k: int = 3, severity_filter: str = None) -> str:
    """Retrieve relevant logs from the knowledge base."""
    try:
        results = logs_store.similarity_search(query, k=k)

        if severity_filter:
            results = [doc for doc in results if doc.metadata.get('severity', '').lower() == severity_filter.lower()]

        if not results:
            return "No relevant logs found."

        context = "\n\n".join([f"[Log {i + 1}]\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Retrieved {len(results)} log(s):\n\n{context}"
    except Exception as e:
        return f"Error retrieving logs: {str(e)}"


@tool
def retrieve_runbook(query: str, k: int = 3, service_filter: str = None) -> str:
    """
    Retrieve runbook sections relevant to the query with citations.

    Args:
        query: Search query describing the issue or alert
        k: Number of sections to return (default: 3)
        service_filter: Optional service name filter

    Returns:
        Formatted runbook sections with citations (runbook_id, section_id, snippet)
    """
    try:
        results = runbook_store.similarity_search(query, k=k)

        if service_filter:
            results = [doc for doc in results if service_filter.lower() in doc.metadata.get('services', '').lower()]

        if not results:
            return f"No relevant runbook sections found{' for service: ' + service_filter if service_filter else ''}."

        citations = []
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            citation = f"""
[Citation {i}]
Runbook ID: {metadata.get('runbook_id', 'N/A')}
Section ID: {metadata.get('section_id', 'N/A')}
Title: {metadata.get('title', 'N/A')}
Section: {metadata.get('section_title', 'N/A')}
Scenario: {metadata.get('scenario', 'N/A')}
Services: {metadata.get('services', 'N/A')}

{doc.page_content}
            """.strip()
            citations.append(citation)

        return f"Retrieved {len(results)} runbook section(s):\n\n" + "\n\n" + ("-" * 80) + "\n\n".join(citations)

    except Exception as e:
        return f"Error retrieving runbook: {str(e)}"

# LangGraph setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([retrieve_alerts, retrieve_traces, retrieve_metrics, retrieve_logs, retrieve_runbook])


def invoke_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", invoke_model)
workflow.add_node("tools", ToolNode([retrieve_alerts, retrieve_traces, retrieve_metrics, retrieve_logs, retrieve_runbook]))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()


def run_query(query: str):
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "retrieved_docs": []
    }
    result = graph.invoke(initial_state)
    return result["messages"][-1].content


# Flask APIs
@app.route('/api/query/alerts', methods=['POST'])
def query_alerts():
    """Query alerts using natural language."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = run_query(f"Search for alerts: {query}")
    return jsonify({"query": query, "response": response})


@app.route('/api/query/traces', methods=['POST'])
def query_traces():
    """Query traces using natural language."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = run_query(f"Search for traces: {query}")
    return jsonify({"query": query, "response": response})


@app.route('/api/query/metrics', methods=['POST'])
def query_metrics():
    """Query metrics using natural language."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = run_query(f"Search for metrics: {query}")
    return jsonify({"query": query, "response": response})


@app.route('/api/query/logs', methods=['POST'])
def query_logs():
    """Query logs using natural language."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = run_query(f"Search for logs: {query}")
    return jsonify({"query": query, "response": response})

@app.route('/api/query/runbook', methods=['POST'])
def query_runbook():
    """Query runbook sections."""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 3)
    service_filter = data.get('service_filter')

    if not query:
        return jsonify({"error": "Query parameter required"}), 400

    result = retrieve_runbook.invoke({
        "query": query,
        "k": k,
        "service_filter": service_filter
    })

    return jsonify({
        "query": query,
        "result": result,
        "k": k,
        "service_filter": service_filter
    })


@app.route('/api/query', methods=['POST'])
def query_all():
    """Query all data sources using natural language."""
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = run_query(query)
    return jsonify({"query": query, "response": response})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


# Initialize data on startup
@app.before_first_request
def initialize():
    """Load all data before first request."""
    load_alerts_data()
    load_traces_data()
    load_metrics_data()
    load_logs_data()


if __name__ == "__main__":
    # Load data
    load_alerts_data()
    load_traces_data()
    load_metrics_data()
    load_logs_data()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)