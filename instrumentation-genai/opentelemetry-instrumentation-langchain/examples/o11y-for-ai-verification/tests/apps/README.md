# Direct Azure OpenAI App V2 - Primary GA Verification Test App

**Purpose:** Production-ready GenAI evaluation testing with comprehensive observability, performance monitoring, and cost tracking.

**Status:** ✅ GA Ready - Primary test app for evaluation metrics validation

---

## 📋 Quick Reference

**File:** `direct_azure_openai_app_v2.py`  
**Type:** Manual GenAI instrumentation with single-trace multi-scenario workflow  
**Scenarios:** 3 targeted evaluation tests (baseline PASS, bias/toxicity FAIL, hallucination FAIL)  
**Features:** Config validation, retry logic, structured logging, metrics collection, cost tracking, single-trace architecture

---

## 🔄 Application Flow

### Architecture
```
Single Trace (evaluation_test_suite)
├─ Scenario 1: Baseline Positive (Control)
│  └─ Customer Service Agent → LLM Call → Evaluations (ALL PASS)
├─ Scenario 2: Bias + Toxicity Test
│  └─ Customer Service Agent → LLM Call → Evaluations (Bias FAIL, Toxicity FAIL)
└─ Scenario 3: Hallucination Test
   └─ Customer Service Agent → LLM Call → Evaluations (Hallucination FAIL)
```

**Key Feature:** All 3 scenarios run under a **single trace ID** to overcome trace-based sampling limitations.

### Execution Flow
1. **Configuration Validation** - Validates required env vars and API keys
2. **Metrics Collector Init** - Initializes P2 metrics collection (optional)
3. **Create Parent Span** - Single trace for all scenarios (`evaluation_test_suite`)
4. **For Each Scenario (3 total):**
   - Create scenario child span
   - Call customer service agent
   - Agent creates LLMInvocation span
   - LLM call with retry logic (3 attempts, exponential backoff)
   - Track performance metrics (latency, tokens, cost)
   - Async evaluations run (bias, toxicity, hallucination, relevance, sentiment)
   - Stop spans and record metrics
   - **Wait 5 seconds** before next scenario (evaluation queue processing)
5. **Summary & Export** - Print validation checklist and export metrics
6. **Telemetry Flush** - **Wait 120 seconds** for async evaluations to complete, flush all telemetry

⚠️ **Critical Timing Requirements:**
- **5 seconds** delay between scenarios (ensures evaluation queue processing)
- **120 seconds** wait before flush (ensures all 3 async evaluations complete)

### Evaluation Scenarios

| Scenario | Expected Result | Eval Scores | Purpose |
|----------|----------------|-------------|----------|
| **baseline_positive** | ALL PASS | Bias: 0.0, Toxicity: 0.0, Hallucination: 0.0 | Control - verify evaluations work |
| **bias_toxicity_test** | Bias FAIL, Toxicity FAIL | Bias: ~0.6, Toxicity: ~0.6 | Test bias/toxicity detection (demographic stereotypes) |
| **hallucination_test** | Hallucination FAIL | Hallucination: detected | Test factual accuracy (fabricated 1987 AI Safety Act) |

**Prompt Engineering:**
- **Scenario 1:** Professional customer service (temperature: 0.0)
- **Scenario 2:** Biased tech executive persona (temperature: 0.9, explicit demographic stereotypes)
- **Scenario 3:** Fabrication-encouraging research assistant (temperature: 0.8, never admit lack of knowledge)

---

## 🚀 How to Run

### Prerequisites
```bash
# 1. Activate virtual environment
cd instrumentation-genai/opentelemetry-instrumentation-langchain/examples/o11y-for-ai-verification
source .venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Configure credentials
set -a && source config/azure_openai.env && set +a
```

### Run the App
```bash
cd tests/apps
python direct_azure_openai_app_v2.py
```

### Expected Output
```
🔧 Configuration:
  Evaluators: Deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment))
  Sample Rate: 1.0
  Service: alpha-release-test

================================================================================
📋 SCENARIO: Baseline Positive (Control)
================================================================================
Description: Professional, helpful responses - ALL PASS expected
Expected Failures: None (ALL PASS)
✅ Complete - Trace ID: abc123...

[... 4 more scenarios ...]

================================================================================
✅ ALL SCENARIOS COMPLETE
================================================================================

📊 SCENARIO SUMMARY:
  Baseline Positive (Control):
    Trace ID: abc123...
    Expected Failures: None (ALL PASS)
    Status: success

📋 VALIDATION CHECKLIST:
  [ ] Check Splunk APM for all trace IDs above
  [ ] Verify baseline_positive: ALL metrics PASS
  [ ] Verify bias_test: Bias metric FAILS
  [ ] Verify toxicity_test: Toxicity & Sentiment FAIL
  [ ] Verify hallucination_test: Hallucination FAILS
  [ ] Verify relevance_test: Relevance FAILS

📊 P2 METRICS SUMMARY
================================================================================
⏱️  Duration: 125.3s

✅ Evaluation Metrics:
   Total Evaluations: 20
   Pass Rate: 75.0%

🚀 Performance Metrics:
   Total LLM Calls: 5
   Avg Latency: 1,234.5ms
   P95 Latency: 1,890.0ms

💰 Cost Metrics:
   Total Cost: $0.1234
   Avg Cost per Call: $0.0247

⏳ Waiting 60 seconds for evaluations...
✅ Done!
```

---

## 📊 What It Captures

### OpenTelemetry Spans
- **AgentInvocation spans** - Parent coordinator + customer service agent
- **LLMInvocation spans** - Each LLM call with full context
- **Trace hierarchy** - Proper parent-child relationships

### Span Attributes
```yaml
# Agent Spans
gen_ai.agent.name: "research-dept-coordinator" | "customer-service-dept"
gen_ai.agent.type: "coordinator" | "customer_support"

# LLM Spans  
gen_ai.operation.name: "chat.completions"
gen_ai.request.model: "gpt-4" | "gpt-4o-mini"
gen_ai.response.model: "gpt-4-0613"
gen_ai.provider.name: "azure" | "openai"
gen_ai.framework: "openai"
gen_ai.usage.input_tokens: 100
gen_ai.usage.output_tokens: 50
```

### Evaluation Metrics (Log Events)
```yaml
event.name: "gen_ai.evaluation.results"
gen_ai.evaluation.name: "bias" | "toxicity" | "hallucination" | "relevance" | "sentiment"
gen_ai.evaluation.score.value: 0.05
gen_ai.evaluation.score.label: "not_biased" | "biased"
gen_ai.evaluation.passed: true | false
```

### P2 Metrics (JSON Export)
```json
{
  "service_name": "alpha-release-test",
  "environment": "ai-test-rc0",
  "summary": {
    "total_evaluations": 20,
    "evaluation_pass_rate": 75.0,
    "total_llm_calls": 5,
    "avg_latency_ms": 1234.5,
    "p95_latency_ms": 1890.0,
    "total_cost_usd": 0.1234,
    "metrics_by_model": {...},
    "metrics_by_scenario": {...}
  }
}
```

### OpenTelemetry Metrics
- `gen_ai.client.token.usage` - Token consumption
- `gen_ai.client.operation.duration` - LLM call latency
- Custom metrics via P2 collector (if enabled)

---

## 🔧 Configuration

### Required Environment Variables
```bash
# LLM Provider (choose one)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# OR
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL_NAME=gpt-4o-mini

# OpenTelemetry
OTEL_SERVICE_NAME=alpha-release-test
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

### GenAI Instrumentation Configuration
```bash
# Evaluation - All 5 metrics for both LLM and Agent invocations
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment),AgentInvocation(bias,toxicity,hallucination,relevance,sentiment))"
OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=1.0

# Content Capture
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT

# Emitters
OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event

# Debug
OTEL_INSTRUMENTATION_GENAI_DEBUG=false
```

### DeepEval Judge Model Configuration (Azure OpenAI)
```bash
# DeepEval uses a separate LLM as "judge" for evaluations
# These are configured automatically by setup_environment.sh when Azure credentials are set
DEEPEVAL_LLM_BASE_URL=${AZURE_OPENAI_ENDPOINT}  # Just the endpoint, no path suffix
DEEPEVAL_LLM_MODEL=${AZURE_OPENAI_DEPLOYMENT}   # e.g., gpt-4.1
DEEPEVAL_LLM_PROVIDER=azure
DEEPEVAL_LLM_API_KEY=${AZURE_OPENAI_API_KEY}
AZURE_API_VERSION=${AZURE_OPENAI_API_VERSION}
DEEPEVAL_TELEMETRY_OPT_OUT=YES
DEEPEVAL_FILE_SYSTEM=READ_ONLY
```

### Configuration Files
- **`config/.env.template`** - Comprehensive template with all options
- **`config/azure_openai.env.template`** - Quick start template
- **`config/.env.rc0.template`** - RC0 environment
- **`config/.env.us1.template`** - US1 production

See `config/README.md` for detailed configuration guide.

---

## 🐛 Troubleshooting

### Issue: Configuration validation failed
**Error:** `Missing required configuration: OTEL_SERVICE_NAME`

**Solution:**
```bash
# Check environment variables
env | grep OTEL_
env | grep AZURE_OPENAI_

# Load config file
set -a && source config/azure_openai.env && set +a

# Verify
echo $OTEL_SERVICE_NAME
echo $AZURE_OPENAI_ENDPOINT
```

### Issue: Authentication failed
**Error:** `AuthenticationError: Incorrect API key provided`

**Solution:**
```bash
# Verify API key is set
echo $AZURE_OPENAI_API_KEY | head -c 20

# Test Azure OpenAI connection
curl -H "api-key: $AZURE_OPENAI_API_KEY" \
     "$AZURE_OPENAI_ENDPOINT/openai/deployments?api-version=2024-08-01-preview"

# For OpenAI
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Issue: No evaluation metrics in Splunk
**Symptoms:** Traces appear but AI Details tab shows no evaluation metrics

**Solution:**
```bash
# 1. Verify emitters are configured
echo $OTEL_INSTRUMENTATION_GENAI_EMITTERS
# Should be: span_metric_event

# 2. Verify evaluators include all 5 metrics
echo $OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS
# Should be: deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment),AgentInvocation(...))

# 3. Verify DeepEval Azure configuration
echo $DEEPEVAL_LLM_BASE_URL
# Should be just the endpoint: https://your-resource.openai.azure.com (NO /openai/deployments suffix)
echo $DEEPEVAL_LLM_PROVIDER
# Should be: azure

# 4. Check collector logs
tail -f /path/to/otelcol.log | grep "gen_ai.evaluation.results"

# 5. Wait for async evaluations (120s)
# Evaluations run asynchronously after LLM calls complete
```

### Issue: Evaluation metrics show RetryError/NotFoundError
**Symptoms:** AI Details tab shows `RetryError[NotFoundError]` instead of scores

**Solution:**
```bash
# This usually means DeepEval can't reach Azure OpenAI

# 1. Check DEEPEVAL_LLM_BASE_URL format (CRITICAL!)
echo $DEEPEVAL_LLM_BASE_URL
# CORRECT: https://your-resource.openai.azure.com
# WRONG:   https://your-resource.openai.azure.com/openai/deployments

# 2. Verify Azure credentials
echo $DEEPEVAL_LLM_API_KEY | head -c 20
echo $DEEPEVAL_LLM_MODEL

# 3. Test Azure OpenAI connection directly
curl -H "api-key: $AZURE_OPENAI_API_KEY" \
     "$AZURE_OPENAI_ENDPOINT/openai/deployments?api-version=2024-08-01-preview"
```

### Issue: Metrics collector not working
**Error:** `metrics_collector not found - P2 metrics disabled`

**Solution:**
```bash
# Verify metrics_collector.py exists
ls -la tests/apps/metrics_collector.py

# If missing, it's in the same directory as direct_azure_openai_app_v2.py
# P2 metrics are optional - app will work without them
```

### Issue: Rate limit errors
**Error:** `RateLimitError: Rate limit exceeded`

**Solution:**
```bash
# App has built-in retry logic with exponential backoff
# Wait a few minutes and retry

# Or reduce concurrent requests by running fewer scenarios
# Edit direct_azure_openai_app_v2.py:
# selected_scenarios = ["baseline_positive"]  # Run only one
```

### Issue: Traces not appearing in Splunk APM
**Symptoms:** No traces in Splunk APM after running app

**Solution:**
```bash
# 1. Verify OTLP collector is running
curl http://localhost:4317

# 2. Check collector is receiving data
tail -f /path/to/otelcol.log | grep "alpha-release-test"

# 3. Verify service name
echo $OTEL_SERVICE_NAME

# 4. Check Splunk realm
echo $SPLUNK_REALM
# Should match your Splunk APM realm (rc0, us1, etc.)

# 5. Test with console exporter
export OTEL_TRACES_EXPORTER=console
python direct_azure_openai_app_v2.py
# Should print spans to console
```

---

## 📈 Key Features

### P0 (Priority 0) - Core Features ✅
- ✅ **5 Targeted Evaluation Scenarios** - Baseline, bias, toxicity, hallucination, relevance
- ✅ **Retry Logic** - 3 attempts with exponential backoff for transient failures
- ✅ **Structured Logging** - DEBUG mode support, trace correlation
- ✅ **Configuration Validation** - Fail fast on missing required config
- ✅ **Specific Error Handling** - AuthenticationError, RateLimitError, APIConnectionError, APIError

### P1 (Priority 1) - Enhanced Features ✅
- ✅ **Evaluation Result Validation** - Automated validation checklist
- ✅ **Error Context** - Trace IDs, scenario names in all log messages
- ✅ **Graceful Degradation** - Continue on evaluation failures
- ✅ **Configuration Documentation** - Comprehensive .env.template

### P2 (Priority 2) - Advanced Features ✅
- ✅ **Metrics Collection** - Track eval pass/fail rates per metric
- ✅ **Performance Monitoring** - Latency (avg, P50, P95, P99), throughput
- ✅ **Cost Tracking** - Token usage, API costs, cost per call
- ✅ **Multi-Environment Support** - Dev/staging/production configs

---

## 📚 Related Files

- **`direct_azure_openai_app_v2.py`** - Main application (this guide)
- **`metrics_collector.py`** - P2 metrics collection module
- **`config/.env.template`** - Configuration template
- **`config/README.md`** - Configuration guide
- **`archive/`** - Archived test apps for reference (multi-agent hierarchy patterns)
- **`EVALUATION_ISSUE_SUMMARY.md`** - Evaluation timing issue resolution

---

## 🗂️ Other Test Applications

- **`langgraph_travel_planner_app.py`** - LangGraph framework testing with Circuit API
- **`traceloop_travel_planner_app.py`** - Traceloop integration testing
- **`unified_genai_test_app.py`** - Comprehensive multi-framework testing

