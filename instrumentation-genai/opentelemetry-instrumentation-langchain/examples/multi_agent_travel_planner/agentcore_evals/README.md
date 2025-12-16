# Deploy LangChain app on Amazon Bedrock AgentCore

This example demonstrates deploying a LangChain multi-agent travel planner to **Amazon Bedrock AgentCore** with Splunk distro of OpenTelemetry instrumentation sending `traces`, `metrics` and `logs` to **Splunk**.

## What is Amazon Bedrock AgentCore?

[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html) is a managed runtime service for hosting and scaling AI agents on AWS. It's **framework and model agnostic** — you can deploy agents built with LangChain, CrewAI, Strands, or custom frameworks.

## Prerequisites

```bash
# Install AWS CLI and AgentCore CLI
# IMPORTANT: bedrock-agentcore-starter-toolkit >= 0.2.5 is required for file permission fix (PR #407)
pip install awscli bedrock-agentcore>=1.1.1 bedrock-agentcore-starter-toolkit>=0.2.5

# Configure AWS credentials
aws configure

# Verify AgentCore access
agentcore --help
```

> **Note:** Version `bedrock-agentcore-starter-toolkit>=0.2.5` includes a critical fix for file permissions in deployment zips ([PR #407](https://github.com/aws/bedrock-agentcore-starter-toolkit/pull/407)). Earlier versions may cause "Permission denied" errors at runtime.

---

## Entrypoint

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict) -> dict:
    """
    AgentCore entrypoint - receives JSON payload, returns JSON response.
    
    Expected payload:
    {
        "origin": "Seattle",
        "destination": "Paris",
        "travellers": 2
    }
    """
    origin = payload.get("origin", "Seattle")
    destination = payload.get("destination", "Paris")
    
    try:
        result = process_request(origin, destination)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    app.run(port=8080)
```

---

## Quick Start

> ```bash
> cp -R instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/agentcore_evals ~/agentcore_evals
> cd ~/agentcore_evals
> ```

### 1. Local Testing

Test the application locally before deploying to AWS:

```bash
# Navigate to the agentcore directory (if not already there)
# Set environment variables
export CISCO_CLIENT_ID=your-client-id
export CISCO_CLIENT_SECRET=your-client-secret
export CISCO_APP_KEY=your-app-key
export OPENAI_API_KEY=your-openai-api-key

# Run locally with AgentCore local server
agentcore deploy --local

# In another terminal, test the endpoint
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "San Francisco",
    "destination": "Tokyo",
    "user_request": "Plan a week-long trip with boutique hotels",
    "travellers": 2
  }'
```

### 2. Deploy to AWS AgentCore

```bash
# Configure the agent (creates .bedrock_agentcore.yaml)
agentcore configure -e main.py

# Deploy to AWS (use --force-rebuild-deps after requirements.txt changes)
# Send traces and metrics to the Splunk's ingestion endpoint using http/protobuf
agentcore deploy --force-rebuild-deps \
  --env OPENAI_API_KEY=<your-openai-api-key> \
  --env OPENAI_MODEL=gpt-4o-mini \
  --env CISCO_CLIENT_ID=<your-cisco-client-id> \
  --env CISCO_CLIENT_SECRET=<your-cisco-client-secret> \
  --env CISCO_APP_KEY=<your-cisco-app-key> \
  --env OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
  --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://ingest.{REALM}.signalfx.com/v2/trace/otlp \
  --env OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://ingest.{REALM}.signalfx.com/v2/datapoint/otlp \
  --env OTEL_EXPORTER_OTLP_HEADERS="X-SF-Token=<your-splunk-token>" \
  --env OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA \
  --env OTEL_SERVICE_NAME=travel-planner-agentcore \
  --env OTEL_RESOURCE_ATTRIBUTES=deployment.environment=aws-agentcore \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk \
  --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
  --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT \
  --env OTEL_INSTRUMENTATION_GENAI_DEBUG=false \
  --env OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults \
  --env OTEL_GENAI_EVAL_DEBUG_SKIPS=true \
  --env OTEL_GENAI_EVAL_DEBUG_EACH=false \
  --env DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300 \
  --env DEEPEVAL_RETRY_MAX_ATTEMPTS=2 \
  --env DEEPEVAL_FILE_SYSTEM=READ_ONLY \
  --env DEEPEVAL_TELEMETRY_OPT_OUT=YES \
  --env CREWAI_DISABLE_TELEMETRY=true \
  --env OTEL_LOGS_EXPORTER=none \
  --env OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true \
  --env DISABLE_ADOT_OBSERVABILITY=true \
  --env HOME=/tmp
```
---

### 3. Invoke the Deployed Agent

```bash
# Via AgentCore CLI
agentcore invoke '{"origin": "New York", "destination": "London", "travellers": 3}'
```

**Example Response:**

```
Response:
{
  "status": "success",
  "session_id": "8852f37d-55d0-48c3-9bd7-c5ca01a809d2",
  "origin": "New York",
  "destination": "London",
  "departure": "2026-01-10",
  "return_date": "2026-01-17",
  "travellers": 3,
  "flight_summary": "SkyLine non-stop service, $727 return in Premium Economy",
  "hotel_summary": "The Atlas near historic centre, $293/night with breakfast",
  "activities_summary": "Tower of London, London Eye, British Museum, West End show...",
  "final_itinerary": "### Week-Long Itinerary: New York to London...",
  "agent_steps": [
    {"agent": "coordinator", "status": "completed"},
    {"agent": "flight_specialist", "status": "completed"},
    {"agent": "hotel_specialist", "status": "completed"},
    {"agent": "activity_specialist", "status": "completed"},
    {"agent": "plan_synthesizer", "status": "completed"}
  ]
}
```

**View logs during invocation:**

```bash
# Follow logs in real-time
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT \
  --log-stream-name-prefix "2025/12/11/[runtime-logs" --follow

# View last hour of logs
aws logs tail /aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT \
  --log-stream-name-prefix "2025/12/11/[runtime-logs" --since 1h
```

**Via AWS CLI:**

```bash
aws bedrock-agentcore-runtime invoke-agent-runtime \
  --agent-runtime-id <your-runtime-id> \
  --payload '{"origin": "Seattle", "destination": "Paris", "travellers": 2}'
```

---

## Sending Telemetry (Traces, Metrics and Logs) to Splunk Observability Cloud

### Splunk OTel Collector Gateway on EKS

Deploy the Splunk Distribution of OpenTelemetry Collector on EKS in the same VPC as AgentCore. This provides centralized telemetry processing, filtering, and forwarding to Splunk Observability Cloud.

#### Configure AgentCore to Send `traces, metrics and logs` to Collector

```bash
# Get NLB endpoint from your OTel Collector service
NLB_DNS=$(kubectl get svc splunk-otel-collector -n splunk-monitoring \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Deploy AgentCore with collector endpoint
agentcore deploy --force-rebuild-deps \
  --env OPENAI_API_KEY=<your-openai-api-key> \
  --env OPENAI_MODEL=gpt-4o-mini \
  --env CISCO_CLIENT_ID=<your-cisco-client-id> \
  --env CISCO_CLIENT_SECRET=<your-cisco-client-secret> \
  --env CISCO_APP_KEY=<your-cisco-app-key> \
  --env OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
  --env OTEL_EXPORTER_OTLP_ENDPOINT=http://${NLB_DNS}:4317 \
  --env OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA \
  --env OTEL_SERVICE_NAME=travel-planner-agentcore \
  --env OTEL_RESOURCE_ATTRIBUTES=deployment.environment=aws-agentcore \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk \
  --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
  --env OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT \
  --env OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true \
  --env OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults \
  --env OTEL_GENAI_EVAL_DEBUG_SKIPS=true \
  --env OTEL_GENAI_EVAL_DEBUG_EACH=false \
  --env DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300 \
  --env DEEPEVAL_RETRY_MAX_ATTEMPTS=2 \
  --env DEEPEVAL_FILE_SYSTEM=READ_ONLY \
  --env DEEPEVAL_TELEMETRY_OPT_OUT=YES \
  --env CREWAI_DISABLE_TELEMETRY=true \
  --env OTEL_LOGS_EXPORTER=otlp \
  --env OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true \
  --env DISABLE_ADOT_OBSERVABILITY=true \
  --env HOME=/tmp
```

> **Note:** When using the OTel Collector, set `OTEL_LOGS_EXPORTER=otlp` since the collector handles log forwarding to Splunk Platform via HEC.

---

## Viewing GenAI Evaluations in Splunk APM

Once the telemetry pipeline is configured correctly, you can view GenAI traces with **evaluation results** directly in Splunk.

### Trace View with Evaluations

![Splunk APM Trace View with GenAI Evaluations](./images/image.png)

The screenshot above shows:

| Feature | Description |
|---------|-------------|
| **Agent Flow** | Visual workflow showing `invoke_workflow` → `flight_specialist`, `activity_specialist`, `hotel_specialist` |
| **Span Properties** | Agent name, operation type, model (`ChatOpenAI`), token counts |
| **Evaluations** | DeepEval results displayed inline: Bias, Toxicity, Relevance, Hallucination, Sentiment |
| **Messages** | Input/output message content captured for each LLM call |

### Evaluation Metrics Displayed

- ✅ **Bias**: Not Biased
- ✅ **Toxicity**: Not Toxic
- ✅ **Relevance**: Pass
- ✅ **Hallucination**: Not Hallucinated
- ✅ **Sentiment**: Positive

---

## Project Structure

```
agentcore_evals/
├── main.py              # LangChain travel planner with AgentCore entrypoint
├── requirements.txt     # Python dependencies
├── images/
│   └── image.png        # Screenshot of Splunk APM with evaluations
├── util/
│   ├── __init__.py
│   └── cisco_token_manager.py  # OAuth2 token management for Cisco LLM
└── README.md            # This file
```

---

## References

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- [AgentCore Samples Repository](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [Splunk OTLP Ingest - General](https://help.splunk.com/en/splunk-observability-cloud/manage-data/other-data-ingestion-methods/other-data-ingestion-methods)
- [Splunk OTLP Metrics Endpoint API](https://dev.splunk.com/observability/reference/api/ingest_data/latest#endpoint-send-otlp-metrics)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)

