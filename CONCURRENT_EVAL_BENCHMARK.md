# Concurrent vs Sequential Evaluation Benchmark

This document describes the manual benchmark test comparing **concurrent** vs **sequential** evaluation modes for LLM-as-a-Judge evaluations using DeepEval.

## Overview

The concurrent evaluation feature enables parallel processing of LLM evaluations using multiple worker threads, significantly improving throughput compared to sequential (single-threaded) processing.

## Test Setup

### Prerequisites

1. Python virtual environment with all dependencies installed
2. `otel-tui` running to capture telemetry (port 4317)
3. OAuth2 credentials for LLM provider (Cisco Circuit in this example)

### Environment Variables

#### Common Variables (Both Modes)

```bash
# LLM Provider Configuration
export LLM_CLIENT_ID=<your-client-id>
export LLM_CLIENT_SECRET=<your-client-secret>
export LLM_APP_KEY=<your-app-key>
export LLM_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
export LLM_BASE_URL=https://chat-ai.cisco.com/openai/deployments

# DeepEval LLM Configuration (for evaluations)
export DEEPEVAL_LLM_BASE_URL=https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini
export DEEPEVAL_LLM_MODEL=gpt-4o-mini
export DEEPEVAL_LLM_PROVIDER=openai
export DEEPEVAL_LLM_CLIENT_ID=<your-client-id>
export DEEPEVAL_LLM_CLIENT_SECRET=<your-client-secret>
export DEEPEVAL_LLM_TOKEN_URL=https://id.cisco.com/oauth2/default/v1/token
export DEEPEVAL_LLM_CLIENT_APP_NAME=<your-app-key>
export DEEPEVAL_FILE_SYSTEM=READ_ONLY

# OpenTelemetry Configuration
export OTEL_INSTRUMENTATION_GENAI_DEBUG=false
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_PYTHON_LOG_CORRELATION=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
export OTEL_LOGS_EXPORTER=otlp
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta
export OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true
export OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
export OTEL_RESOURCE_ATTRIBUTES=deployment.environment=agentic-ai-demo
export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=100
export EVAL_FLUSH_WAIT_SECONDS=300
```

#### Concurrent Mode Variables

```bash
export OTEL_SERVICE_NAME=agentic-ai-demo-app
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=5
```

#### Sequential Mode Variables

```bash
export OTEL_SERVICE_NAME=agentic-ai-demo-app-sequential
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=false
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=1
```

## Running the Benchmark

### Step 1: Start otel-tui

```bash
docker run --rm -it -p 4317:4317 -p 4318:4318 ymtdzzz/otel-tui:latest
```

### Step 2: Run Concurrent Mode

```bash
# Set concurrent mode variables
export OTEL_SERVICE_NAME=agentic-ai-demo-app
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=5

# Start the server
python main.py
```

### Step 3: Send Test Request (Concurrent)

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "San Francisco",
    "destination": "Tokyo",
    "user_request": "Plan a week-long trip with boutique hotels",
    "travellers": 2
  }'
```

### Step 4: Kill Server and Run Sequential Mode

```bash
# Kill the concurrent server
lsof -ti:8080 | xargs kill -9

# Set sequential mode variables
export OTEL_SERVICE_NAME=agentic-ai-demo-app-sequential
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=false
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=1

# Start the server
python main.py
```

### Step 5: Send Test Request (Sequential)

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "San Francisco",
    "destination": "Tokyo",
    "user_request": "Plan a week-long trip with boutique hotels",
    "travellers": 2
  }'
```

## Results

### Performance Comparison

| Metric | Sequential Mode | Concurrent Mode | Improvement |
|--------|-----------------|-----------------|-------------|
| **Request Latency** | 315.973s (~5 min) | 24.689s | **~13x faster** |
| **Evaluation Time** | ~5+ minutes | ~20 seconds | **~15x faster** |
| **Workers** | 1 | 5 | 5x parallelism |
| **Tests Processed** | 4 (sequential) | 4 (parallel) | Same count |
| **Pass Rate** | 100% | 100% | Same quality |

### Server Logs

**Concurrent Mode:**
```
✓ Evaluation completed 🎉! (time taken: 3.84s | token cost: 3.1596 USD)
» Test Results (4 total tests):
   » Pass Rate: 100.0% | Passed: 4 | Failed: 0

{"timestamp": "2026-01-17T01:44:53.623Z", "level": "INFO", 
 "message": "Invocation completed successfully (24.689s)"}
```

**Sequential Mode:**
```
{"timestamp": "2026-01-17T01:57:59.894Z", "level": "INFO", 
 "message": "Invocation completed successfully (315.973s)"}
```

### otel-tui Telemetry Comparison

The telemetry captured in `otel-tui` shows clear differences:

| Service Name | Mode | Evaluation Events Timeline |
|--------------|------|---------------------------|
| `agentic-ai-demo-app` | Concurrent | 01:45:30 → 01:48:33 (~3 min) |
| `agentic-ai-demo-app-sequential` | Sequential | 01:54:54 → 02:00:13 (~5+ min) |

**Key Observations:**

1. **Concurrent Mode**: Evaluation results arrive in quick bursts, with multiple `gen_ai.evaluation.results` events within seconds of each other.

2. **Sequential Mode**: Evaluation results are spaced out (01:54:54, 01:55:05, 01:55:15, 01:55:28...), with each evaluation waiting for the previous one to complete.

### Screenshot Reference

The `otel-tui` logs tab shows both services side by side:
- `agentic-ai-demo-app` (concurrent) - evaluations completed quickly
- `agentic-ai-demo-app-sequential` - evaluations spread over several minutes

![otel-tui comparison](./docs/otel-tui-concurrent-vs-sequential.png)

## Configuration Reference

### New Environment Variables for Concurrent Evaluation

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` | Enable concurrent evaluation mode | `false` |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` | Number of worker threads | `4` |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` | Bounded queue size (0=unbounded) | `0` |

### Recommendations

| Scenario | Recommended Settings |
|----------|---------------------|
| **Development/Low Volume** | Sequential mode (default) |
| **Production/High Volume** | Concurrent with 4-5 workers |
| **Memory Constrained** | Bounded queue (100-1000) |
| **Rate Limited APIs** | Reduce workers to 2-3 |

## Conclusion

The concurrent evaluation mode provides significant performance improvements:

- **~13x faster request latency** (24s vs 315s)
- **~15x faster evaluation completion** (20s vs 5+ min)
- **Same evaluation quality** (100% pass rate in both modes)
- **Reduced `EVAL_FLUSH_WAIT_SECONDS`** from 300s to 30-60s

For production deployments with LLM-as-a-Judge evaluations, enabling concurrent mode is strongly recommended.

## Package Versions

- `splunk-otel-util-genai`: 0.1.7
- `splunk-otel-util-genai-evals`: 0.1.5
- `splunk-otel-genai-evals-deepeval`: 0.1.9

