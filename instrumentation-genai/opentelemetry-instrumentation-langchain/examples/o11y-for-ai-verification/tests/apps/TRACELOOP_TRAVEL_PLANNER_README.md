# Traceloop Travel Planner App

## Overview

**Test Case**: TC-PI2-TRACELOOP-01 - Traceloop SDK Attribute Translation Validation

A multi-agent travel planning application that demonstrates **zero-code instrumentation** using the Traceloop SDK with automatic attribute translation to OpenTelemetry GenAI semantic conventions.

This application validates that the Traceloop translator correctly converts `traceloop.*` attributes to `gen_ai.*` semantic conventions without requiring code changes to the application.

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Traceloop Travel Planner                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │   User      │───▶│   @workflow      │───▶│   LangGraph StateGraph  │    │
│  │   Request   │    │   Decorator      │    │   (Multi-Agent Flow)    │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘    │
│                                                      │                      │
│                     ┌────────────────────────────────┼──────────────────┐   │
│                     │                                ▼                  │   │
│                     │  ┌──────────────────────────────────────────┐    │   │
│                     │  │           Coordinator Agent              │    │   │
│                     │  │         @task(name="coordinator")        │    │   │
│                     │  └──────────────────────────────────────────┘    │   │
│                     │                     │                            │   │
│         ┌───────────┼─────────────────────┼────────────────────────┐   │   │
│         │           │                     │                        │   │   │
│         ▼           ▼                     ▼                        ▼   │   │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐ ┌──────────────┐    │   │
│  │  Flight    │ │   Hotel    │ │   Activity     │ │    Plan      │    │   │
│  │ Specialist │ │ Specialist │ │  Specialist    │ │ Synthesizer  │    │   │
│  │  @task     │ │   @task    │ │    @task       │ │   @task      │    │   │
│  └────────────┘ └────────────┘ └────────────────┘ └──────────────┘    │   │
│        │              │               │                  │            │   │
│        ▼              ▼               ▼                  ▼            │   │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐                     │   │
│  │mock_search │ │mock_search │ │ mock_search    │                     │   │
│  │ _flights   │ │  _hotels   │ │  _activities   │                     │   │
│  │   @tool    │ │   @tool    │ │     @tool      │                     │   │
│  └────────────┘ └────────────┘ └────────────────┘                     │   │
│                                                                       │   │
│                     LangGraph Workflow Nodes                          │   │
└───────────────────────────────────────────────────────────────────────┘   │
                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Telemetry Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Telemetry Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │  Traceloop SDK  │  Creates spans with traceloop.* attributes             │
│  │  @workflow      │  - traceloop.workflow.name                             │
│  │  @task          │  - traceloop.entity.input/output                       │
│  └────────┬────────┘  - traceloop.span.kind                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              TraceloopSpanProcessor (Zero-Code Translator)          │   │
│  │                                                                     │   │
│  │  1. Intercepts Traceloop spans on_end()                            │   │
│  │  2. Extracts traceloop.entity.input/output JSON                    │   │
│  │  3. Reconstructs LangChain message objects (HumanMessage, etc.)    │   │
│  │  4. Creates LLMInvocation with input_messages/output_messages      │   │
│  │  5. Emits synthetic gen_ai.* spans via TelemetryHandler            │   │
│  │  6. Triggers evaluation callbacks (DeepEval)                       │   │
│  └────────┬────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐     │
│  │  Original Span  │    │ Translated Span │    │  Evaluation Events  │     │
│  │ ChatOpenAI.chat │    │ chat gpt-4o-mini│    │ gen_ai.evaluation   │     │
│  │ (traceloop.*)   │    │ (gen_ai.*)      │    │ .results            │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────────┘     │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                              │
│                    │    OTel Collector       │                              │
│                    │    (localhost:4318)     │                              │
│                    └────────────┬────────────┘                              │
│                                 │                                           │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │   Splunk Observability  │                              │
│                    │   - APM (Traces)        │                              │
│                    │   - AI Details Tab      │                              │
│                    │   - Log Observer        │                              │
│                    └─────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dual Span Architecture

The application produces **two types of spans** for each LLM call:

| Span Name | Source | Attributes | AI Details Tab |
|-----------|--------|------------|----------------|
| `ChatOpenAI.chat` | Original Traceloop SDK | `traceloop.*` | Basic info only |
| `chat gpt-4o-mini` | Translated gen_ai span | `gen_ai.*` | Full messages + evaluations |

**Note**: This is expected behavior. The translator creates new synthetic spans with `gen_ai.*` attributes alongside the original Traceloop spans. For AI observability in Splunk, use the `chat gpt-4o-mini` spans (translated ones).

---

## Agents

| Agent | Role | Tools | Temperature |
|-------|------|-------|-------------|
| **Coordinator** | Extract key details, plan workflow | None | 0.2 |
| **Flight Specialist** | Search and recommend flights | `mock_search_flights` | 0.4 |
| **Hotel Specialist** | Search and recommend hotels | `mock_search_hotels` | 0.5 |
| **Activity Specialist** | Curate signature activities | `mock_search_activities` | 0.6 |
| **Plan Synthesizer** | Combine insights into final itinerary | None | 0.3 |

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | `sk-...` |

### OpenTelemetry Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name for traces | `travel-planner-traceloop` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint (HTTP) | `http://localhost:4318` |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes | `""` |

### GenAI Instrumentation

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitter configuration | `span_metric_event,splunk` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION` | Evaluation emitter | `replace-category:SplunkEvaluationResults` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture message content | `true` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | Content capture mode | `SPAN_AND_EVENT` |

### Evaluation Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | Evaluator configuration | `Deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment),AgentInvocation(...))` |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` | Evaluation sampling rate (0.0-1.0) | `1.0` |
| `OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS` | Completion callbacks | `evaluations` |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | Disable DeepEval telemetry | `YES` (recommended) |

---

## Evaluation Metrics

The application uses **DeepEval** for LLM output evaluation. Five metrics are evaluated for each LLM invocation:

| Metric | Description | Pass Criteria |
|--------|-------------|---------------|
| **Bias** | Detects biased or discriminatory content | Score = 0.00 (Not Biased) |
| **Toxicity** | Detects toxic, harmful, or offensive content | Score = 0.00 (Not Toxic) |
| **Hallucination** | Detects fabricated or unsubstantiated information | Factual alignment with input |
| **Relevance** | Measures response relevance to the input | Score = 1.00 (Fully Relevant) |
| **Sentiment** | Analyzes emotional tone of the response | Positive/Neutral/Negative |

### Evaluation Results in Splunk

Evaluation results appear in the **AI Details tab** under "Metrics" section:

```
evaluations
Hover on each metric to view it's details.
├── Bias : Not Biased
│   The score is 0.00 because the actual output is completely unbiased...
├── Toxicity : Not Toxic
│   The score is 0.00 because the output contains no toxic elements...
├── Hallucination : Not Hallucinated
│   The response effectively extracts and organizes all key details...
├── Relevance : Pass
│   The score is 1.00 because the response directly addressed...
└── Sentiment : Positive
    The response effectively captures the positive sentiment...
```

---

## How to Run

### Prerequisites

1. **Python 3.10+**
2. **OpenAI API Key**
3. **OTel Collector** running on `localhost:4318`

### Setup

```bash
# Navigate to the verification directory
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/o11y-for-ai-verification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the util packages in development mode
pip install -e ../../../../util/opentelemetry-util-genai
pip install -e ../../../../util/opentelemetry-util-genai-evals
pip install -e ../../../../util/opentelemetry-util-genai-evals-deepeval
pip install -e ../../../../util/opentelemetry-util-genai-emitters-splunk
pip install -e ../../../../util/opentelemetry-util-genai-traceloop-translator
```

### Run with Environment Setup Script

```bash
# Source the environment setup script
source scripts/setup_environment.sh

# Run the application
cd tests/apps
python traceloop_travel_planner_app.py
```

### Run Manually

```bash
# Set required environment variables
export OPENAI_API_KEY="your-api-key"
export OTEL_SERVICE_NAME="travel-planner-traceloop"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Optional: Disable DeepEval telemetry
export DEEPEVAL_TELEMETRY_OPT_OUT=YES

# Run the application
python traceloop_travel_planner_app.py
```

### Expected Output

```
================================================================================
TRACELOOP TRAVEL PLANNER - Multi-Agent Workflow
================================================================================

🔧 Configuration:
  Service Name: alpha-test-unified-app
  Evaluators: deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment),...)
  Sample Rate: 1.0
  Emitters: span_metric_event,splunk
  Content Capture: true
================================================================================

🌍 Multi-Agent Travel Planner (Traceloop SDK)
============================================================

🤖 Coordinator Agent
[Response content...]

🤖 Flight Specialist Agent
[Response content...]

🤖 Hotel Specialist Agent
[Response content...]

🤖 Activity Specialist Agent
[Response content...]

🤖 Plan Synthesizer Agent
[Response content...]

🎉 Final itinerary
----------------------------------------
[Final travel plan...]

================================================================================
✅ WORKFLOW COMPLETE
================================================================================
🔍 Trace ID: fac59c5efd3df301df9171981b40839e

[SUCCESS] Workflow completed
[SUCCESS] Traces exported with traceloop.* attributes
[SUCCESS] Zero-code translator converted to gen_ai.* attributes

Validation Checklist:
  [ ] Trace ID visible in Splunk APM: fac59c5efd3df301df9171981b40839e
  [ ] AI Details tab shows evaluation metrics
  [ ] gen_ai.* attributes present (translated from traceloop.*)
  [ ] All 5 agents visible in trace hierarchy
```

---

## Debugging

### Enable Debug Logging

The application has debug logging enabled by default for key modules:

```python
logging.getLogger("opentelemetry.util.genai.processor.traceloop_span_processor").setLevel(logging.DEBUG)
logging.getLogger("opentelemetry.util.genai.handler").setLevel(logging.DEBUG)
```

### Debug Environment Variables

```bash
# Enable GenAI debug mode
export OTEL_INSTRUMENTATION_GENAI_DEBUG=true

# View detailed evaluation logs
export DEEPEVAL_VERBOSE=true
```

### Common Issues

#### 1. "Error loading events" in AI Details Tab

**Cause**: Missing `EventLoggerProvider` configuration.

**Solution**: Ensure the app configures both `LoggerProvider` and `EventLoggerProvider`:

```python
from opentelemetry._events import set_event_logger_provider
from opentelemetry.sdk._events import EventLoggerProvider
set_event_logger_provider(EventLoggerProvider(logger_provider=logger_provider))
```

#### 2. Evaluation Metrics Not Appearing

**Cause**: Missing Splunk emitters configuration.

**Solution**: Set these environment variables BEFORE any OpenTelemetry imports:

```python
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span_metric_event,splunk")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION", "replace-category:SplunkEvaluationResults")
```

#### 3. Messages Not Showing in AI Details

**Cause**: Content capture not enabled.

**Solution**: Enable content capture:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
```

#### 4. gRPC Endpoint Error

**Cause**: Traceloop SDK requires HTTP endpoint (port 4318), not gRPC (port 4317).

**Solution**: The app automatically converts:

```python
if ":4317" in OTEL_EXPORTER_OTLP_ENDPOINT:
    OTEL_EXPORTER_OTLP_ENDPOINT = OTEL_EXPORTER_OTLP_ENDPOINT.replace(":4317", ":4318")
```

#### 5. Telemetry Not Flushing

**Cause**: Application exits before async evaluations complete.

**Solution**: The app includes a `flush_telemetry()` function that:
1. Waits for evaluations to complete (up to 200s)
2. Flushes traces, logs, and metrics
3. Waits 5s for batch processors

### Viewing Logs in OTel Collector

```bash
# Check collector logs for evaluation events
tail -f bin/otelcol.log | grep -E "gen_ai.evaluation|SplunkEvaluation"
```

---

## Validation Checklist

After running the application, verify in Splunk:

- [ ] **Trace ID visible in Splunk APM**
  - Search: `sf_service:alpha-test-unified-app` or your service name
  - Filter by the trace ID from the output

- [ ] **AI Details tab shows evaluation metrics**
  - Click on a `chat gpt-4o-mini` span (not `ChatOpenAI.chat`)
  - View "AI details" tab
  - Verify "Metrics" section shows all 5 evaluations

- [ ] **gen_ai.* attributes present**
  - Check span attributes include `gen_ai.request.model`, `gen_ai.operation.name`, etc.

- [ ] **All 5 agents visible in trace hierarchy**
  - Coordinator → Flight Specialist → Hotel Specialist → Activity Specialist → Plan Synthesizer

- [ ] **Messages visible in AI Details**
  - Input and output messages should be displayed
  - Both "Parsed" and "JSON" views available

---

## Key Code Sections

### Traceloop SDK Initialization

```python
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(
    disable_batch=True,
    api_endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
    app_name=OTEL_SERVICE_NAME,
    resource_attributes=resource_attributes,
)
```

### OTLP Logging Configuration

```python
def _configure_otlp_logging() -> None:
    # Configure LoggerProvider for OTLP export
    logger_provider = LoggerProvider(resource=resource)
    log_processor = BatchLogRecordProcessor(OTLPLogExporter(endpoint=log_endpoint))
    logger_provider.add_log_record_processor(log_processor)
    set_logger_provider(logger_provider)
    
    # CRITICAL: Configure EventLoggerProvider for Splunk evaluation emitter
    from opentelemetry._events import set_event_logger_provider
    from opentelemetry.sdk._events import EventLoggerProvider
    set_event_logger_provider(EventLoggerProvider(logger_provider=logger_provider))
```

### Agent Definition with @task Decorator

```python
@task(name="coordinator_agent")
def coordinator_node(state: PlannerState) -> PlannerState:
    llm = _create_llm("coordinator", temperature=0.2, session_id=state["session_id"])
    # ... agent logic
    return state
```

### Workflow Definition with @workflow Decorator

```python
@workflow(name="travel_planner_multi_agent")
def main() -> str:
    # ... workflow logic
    return trace_id
```

---

## Related Files

| File | Description |
|------|-------------|
| `direct_azure_openai_app_v2.py` | Reference app with similar features (Azure OpenAI) |
| `langgraph_travel_planner_app.py` | Similar app without Traceloop (uses LangChain instrumentation) |
| `../../../scripts/setup_environment.sh` | Environment setup script |
| `../../../config/.env.template` | Environment variable template |

---

## References

- [Traceloop SDK Documentation](https://traceloop.com/docs)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Splunk Observability Cloud](https://docs.splunk.com/observability)
- [DeepEval Documentation](https://docs.confident-ai.com/)
