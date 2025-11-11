# Alpha Release Testing Plan - AI Observability Features

## Overview
Comprehensive testing plan for Alpha release features based on customer-facing documentation. This plan covers all instrumentation methods, configuration options, and UI verification for AI monitoring in Splunk Observability Cloud.

---

## Test Environment Setup

### Prerequisites
- **Environment**: lab0 tenant (Splunk Observability Cloud)
- **Python Version**: 3.8+
- **OpenTelemetry SDK**: >= 1.38.0
- **Required Packages**:
  ```bash
  pip install splunk-otel-util-genai
  pip install splunk-otel-genai-emitters-splunk
  pip install splunk-otel-genai-evals-deepeval
  pip install opentelemetry-instrumentation-langchain
  pip install langchain langchain-openai
  pip install traceloop-sdk>=0.47.4  # For Traceloop tests
  ```

### Environment Variables Base Configuration
```bash
# Core OTEL Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=alpha-ai-test
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=lab0-alpha

# GenAI Instrumentation
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

---

## Test Categories

## 1. Instrument AI Applications (Overview)

### Test Case 1.1: Zero-Code vs Code-Based Instrumentation
**Objective**: Verify distinction between zero-code and code-based instrumentation

**Test Steps**:
1. **Zero-Code Test**:
   ```bash
   opentelemetry-instrument \
     --traces_exporter otlp \
     --metrics_exporter otlp \
     python azure_openai_basic.py
   ```
   - Verify traces/metrics sent without code changes
   - Check telemetry in Splunk APM

2. **Code-Based Test**:
   ```python
   from opentelemetry.instrumentation.langchain import LangchainInstrumentor
   LangchainInstrumentor().instrument()
   ```
   - Verify explicit instrumentation works
   - Compare telemetry with zero-code approach

**Expected Results**:
- ✅ Both methods generate traces and metrics
- ✅ Telemetry appears in Splunk APM
- ✅ No code changes required for zero-code

**Test File**: `tests/test_instrumentation_methods.py`

---

## 2. Instrument LangChain/LangGraph Application

### Test Case 2.1: Prerequisites Verification
**Objective**: Verify all required packages install correctly

**Test Steps**:
```bash
# Verify OpenTelemetry SDK version
python -c "import opentelemetry; print(opentelemetry.__version__)"

# Verify package installations
pip list | grep -E "splunk-otel|opentelemetry|langchain"
```

**Expected Results**:
- ✅ opentelemetry-sdk >= 1.38.0
- ✅ All splunk-otel packages installed
- ✅ No dependency conflicts

**Test File**: `tests/test_prerequisites.py`

---

### Test Case 2.2: Zero-Code LangChain Instrumentation
**Objective**: Verify automatic instrumentation of LangChain applications

**Configuration**:
```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
```

**Test Steps**:
1. Deploy simple LangChain app with zero-code instrumentation
2. Execute various prompts (simple, complex, multi-turn)
3. Verify telemetry in Splunk APM

**Expected Results**:
- ✅ Traces generated automatically
- ✅ Metrics sent to Splunk
- ✅ No code modifications required

**Test File**: `tests/test_langchain_zero_code.py`

---

### Test Case 2.3: Code-Based LangChain Instrumentation
**Objective**: Verify explicit LangchainInstrumentor usage

**Test Code**:
```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

# Instrument
LangchainInstrumentor().instrument()

# Create LangChain app
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(...)
result = llm.invoke("Test prompt")
```

**Expected Results**:
- ✅ Traces generated with gen_ai.* attributes
- ✅ Metrics sent to Splunk
- ✅ Proper span hierarchy

**Test File**: `tests/test_langchain_code_based.py`

---

### Test Case 2.4: Agent Name and Workflow Name Configuration
**Objective**: Verify agent_name and workflow_name attributes

**Test Code**:
```python
from langchain.agents import create_agent

agent = create_agent(
    name="weather-agent",  # Sets gen_ai.agent.name
    model=llm,
    tools=[get_weather]
)

# For workflows
workflow = StateGraph(...)
workflow.name = "booking-workflow"  # Sets gen_ai.workflow.name
```

**Test Steps**:
1. Set agent_name for Chains
2. Set workflow_name for Graphs
3. Verify attributes in telemetry

**Expected Results**:
- ✅ `gen_ai.agent.name` appears in spans
- ✅ `gen_ai.workflow.name` appears in spans
- ✅ Entities promoted to AgentInvocation/Workflow
- ✅ Visible in Splunk APM Agents page

**Test File**: `tests/test_agent_workflow_names.py`

---

### Test Case 2.5: Send Evaluation Results (LangChain)
**Objective**: Verify evaluation results sent to Splunk

**Configuration**:
```bash
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
export OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true
export DEEPEVAL_FILE_SYSTEM=READ_ONLY
export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS=deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment))
```

**Test Steps**:
1. Configure evaluation environment variables
2. Run LangChain app with various prompts
3. Verify evaluation results in Splunk

**Expected Results**:
- ✅ Evaluation metrics sent (bias, toxicity, etc.)
- ✅ Results aggregated correctly
- ✅ Visible in Splunk APM AI details tab
- ✅ Quality scores displayed

**Test File**: `tests/test_langchain_evaluations.py`

---

## 3. Instrument Python AI Application (Code-Based)

### Test Case 3.1: Prerequisites for Direct AI Apps
**Objective**: Verify SDK and package compatibility

**Test Steps**:
```bash
pip install splunk-otel-util-genai
python -c "from opentelemetry.util.genai import LLMInvocation; print('Success')"
```

**Expected Results**:
- ✅ opentelemetry-sdk >= 1.38.0
- ✅ splunk-otel-util-genai installed
- ✅ LLMInvocation importable

**Test File**: `tests/test_direct_ai_prerequisites.py`

---

### Test Case 3.2: LLMInvocation for Azure OpenAI
**Objective**: Verify LLMInvocation telemetry for direct Azure OpenAI calls

**Test Code**:
```python
from opentelemetry.util.genai import LLMInvocation
from openai import AzureOpenAI

client = AzureOpenAI(...)

with LLMInvocation(
    request_model="gpt-4",
    provider="azure",
    framework="openai",
    operation="chat.completions"
) as llm_call:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    llm_call.set_input_messages([{"role": "user", "content": "Hello"}])
    llm_call.set_output_messages([{"role": "assistant", "content": response.choices[0].message.content}])
    llm_call.set_token_usage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )
```

**Expected Results**:
- ✅ Span created with gen_ai.* attributes
- ✅ `gen_ai.request.model` = "gpt-4"
- ✅ `gen_ai.provider.name` = "azure"
- ✅ `gen_ai.operation.name` = "chat.completions"
- ✅ Input/output messages captured
- ✅ Token usage recorded

**Test File**: `tests/test_llm_invocation.py`

---

### Test Case 3.3: AgentInvocation for Direct AI Apps
**Objective**: Verify AgentInvocation telemetry

**Test Code**:
```python
from opentelemetry.util.genai import AgentInvocation

with AgentInvocation(
    agent_name="custom-agent",
    provider="azure"
) as agent_call:
    # Execute agent logic
    result = execute_agent_workflow()
    agent_call.set_output(result)
```

**Expected Results**:
- ✅ Span created with agent.* attributes
- ✅ `gen_ai.agent.name` set correctly
- ✅ Promoted to AgentInvocation entity
- ✅ Visible in Splunk APM Agents page

**Test File**: `tests/test_agent_invocation.py`

---

### Test Case 3.4: Send Evaluation Results (Direct AI)
**Objective**: Verify evaluation results for direct AI applications

**Configuration**:
```bash
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
export OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION=true
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric
export OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION=replace-category:SplunkEvaluationResults
export OTEL_GENAI_EVAL_DEBUG_SKIPS=true
export OTEL_GENAI_EVAL_DEBUG_EACH=true
export OTEL_INSTRUMENTATION_GENAI_DEBUG=true
```

**Test Steps**:
1. Configure evaluation settings
2. Run direct AI app with evaluations
3. Check debug logs for skips and results
4. Verify in Splunk APM

**Expected Results**:
- ✅ Evaluation results sent
- ✅ Debug logs show skips
- ✅ Debug logs show each result
- ✅ Results visible in Splunk

**Test File**: `tests/test_direct_ai_evaluations.py`

---

## 4. Collect Data from Traceloop-Instrumented Applications

### Test Case 4.1: Traceloop Prerequisites
**Objective**: Verify Traceloop translator installation

**Test Steps**:
```bash
pip install splunk-otel-util-genai-translator-traceloop
pip install traceloop-sdk>=0.47.4
export DEEPEVAL_TELEMETRY_OPT_OUT="YES"
```

**Expected Results**:
- ✅ Translator installed successfully
- ✅ Traceloop SDK compatible
- ✅ DeepEval telemetry disabled

**Test File**: `tests/test_traceloop_prerequisites.py`

---

### Test Case 4.2: Traceloop Attribute Translation
**Objective**: Verify automatic translation of traceloop.* to gen_ai.*

**Test Code**:
```python
from traceloop.sdk import Traceloop

Traceloop.init(app_name="test-app")

# Run Traceloop-instrumented app
# Verify attributes are translated
```

**Expected Translations**:
- `traceloop.entity.name` → `gen_ai.agent.name`
- `traceloop.workflow.name` → `gen_ai.workflow.name`
- `traceloop.association.properties.*` → `gen_ai.*`

**Verification**:
1. Check spans in Splunk APM
2. Verify gen_ai.* attributes present
3. Confirm no traceloop.* attributes in final spans

**Expected Results**:
- ✅ Automatic translation works
- ✅ gen_ai.* attributes present
- ✅ Traceloop attributes removed

**Test File**: `tests/test_traceloop_translation.py`

---

## 5. Configuration Settings Testing

### Test Case 5.1: OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE
**Objective**: Verify metric temporality options

**Test Configurations**:
```bash
# Test 1: DELTA
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=DELTA

# Test 2: CUMULATIVE
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=CUMULATIVE

# Test 3: LOWMEMORY
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=LOWMEMORY
```

**Expected Results**:
- ✅ DELTA: Metrics show incremental values
- ✅ CUMULATIVE: Metrics show cumulative values
- ✅ LOWMEMORY: Optimized memory usage
- ✅ Correct temporality in Splunk

**Test File**: `tests/test_metric_temporality.py`

---

### Test Case 5.2: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
**Objective**: Verify message content capture control

**Test Configurations**:
```bash
# Test 1: Enabled
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# Test 2: Disabled
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false
```

**Expected Results**:
- ✅ true: Message content in spans/events
- ✅ false: No message content captured
- ✅ Privacy control working

**Test File**: `tests/test_message_content_capture.py`

---

### Test Case 5.3: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE
**Objective**: Verify message content location options

**Test Configurations**:
```bash
# Test 1: NO_CONTENT
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=NO_CONTENT

# Test 2: SPAN_AND_EVENT
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT

# Test 3: SPAN_ONLY
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_ONLY

# Test 4: EVENT_ONLY
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=EVENT_ONLY
```

**Expected Results**:
- ✅ NO_CONTENT: No messages anywhere
- ✅ SPAN_AND_EVENT: Messages in both locations
- ✅ SPAN_ONLY: Messages only in span attributes
- ✅ EVENT_ONLY: Messages only in events

**Test File**: `tests/test_message_content_mode.py`

---

### Test Case 5.4: OTEL_INSTRUMENTATION_GENAI_EMITTERS
**Objective**: Verify telemetry emitter options

**Test Configurations**:
```bash
# Test 1: span only
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span

# Test 2: span + metric
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric

# Test 3: span + metric + event
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event

# Test 4: span + metric + event + splunk
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event,splunk
```

**Expected Results**:
- ✅ span: Only traces generated
- ✅ span_metric: Traces + metrics
- ✅ span_metric_event: Traces + metrics + events
- ✅ splunk: Splunk-specific emitters enabled

**Test File**: `tests/test_emitters.py`

---

### Test Case 5.5: OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE
**Objective**: Verify evaluation sampling

**Test Configurations**:
```bash
# Test 1: 10% sampling
export OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=0.1

# Test 2: 50% sampling
export OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=0.5

# Test 3: 100% sampling
export OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE=1.0
```

**Test Steps**:
1. Run 100 AI calls with each sampling rate
2. Count evaluation results
3. Verify sampling percentage

**Expected Results**:
- ✅ 0.1: ~10 evaluations out of 100
- ✅ 0.5: ~50 evaluations out of 100
- ✅ 1.0: 100 evaluations out of 100
- ✅ Cost optimization working

**Test File**: `tests/test_evaluation_sampling.py`

---

### Test Case 5.6: Debug Configuration
**Objective**: Verify debug logging options

**Test Configurations**:
```bash
export OTEL_INSTRUMENTATION_GENAI_DEBUG=true
export OTEL_GENAI_EVAL_DEBUG_SKIPS=true
export OTEL_GENAI_EVAL_DEBUG_EACH=true
```

**Expected Results**:
- ✅ Debug logs generated
- ✅ Skipped evaluations logged
- ✅ Each evaluation result logged
- ✅ Helpful for troubleshooting

**Test File**: `tests/test_debug_logging.py`

---

## 6. Splunk APM UI Verification

### Test Case 6.1: Agents Page
**Objective**: Verify Agents page in Splunk APM

**Test Steps**:
1. Navigate to APM → Agents
2. Verify page loads correctly
3. Check aggregate metrics display

**Expected Results**:
- ✅ Agents page exists under APM
- ✅ Aggregate metrics shown:
  - Total requests
  - Error rate
  - Latency (P50, P90, P99)
  - Token usage
  - Quality trends
- ✅ Table lists all instrumented agents
- ✅ Individual agent metrics visible:
  - RED metrics (Rate, Errors, Duration)
  - Token usage
  - Estimated cost
  - Quality issues count

**Test File**: `tests/ui/test_agents_page.py` (Playwright)

---

### Test Case 6.2: Agent Filtering and Sorting
**Objective**: Verify filtering and sorting on Agents page

**Test Steps**:
1. Apply filters (by environment, provider, model)
2. Sort by different columns
3. Search for specific agents

**Expected Results**:
- ✅ Filters work correctly
- ✅ Sorting functions properly
- ✅ Search finds agents
- ✅ UI responsive

**Test File**: `tests/ui/test_agents_filtering.py` (Playwright)

---

### Test Case 6.3: Related Traces Navigation
**Objective**: Verify "Related traces" icon functionality

**Test Steps**:
1. Click "Related traces" icon for an agent
2. Verify navigation to Trace Analyzer
3. Check filters applied

**Expected Results**:
- ✅ Navigates to Trace Analyzer
- ✅ Filtered by agent name
- ✅ "AI traces only" filter applied
- ✅ Correct traces displayed

**Test File**: `tests/ui/test_related_traces.py` (Playwright)

---

### Test Case 6.4: Related Logs Navigation
**Objective**: Verify "Related logs" icon functionality

**Test Steps**:
1. Click "Related logs" icon for an agent
2. Verify navigation to Log Observer
3. Check filters applied

**Expected Results**:
- ✅ Navigates to Log Observer
- ✅ Filtered by agent name
- ✅ AI call logs displayed
- ✅ Trace/span correlation visible

**Test File**: `tests/ui/test_related_logs.py` (Playwright)

---

### Test Case 6.5: Agent Detail View
**Objective**: Verify individual agent detail page

**Test Steps**:
1. Click agent name in table
2. Navigate to detail view
3. Verify all charts and data

**Expected Results**:
- ✅ Detail view loads correctly
- ✅ Charts display:
  - Request rate over time
  - Error rate over time
  - Latency percentiles
  - Token usage trends
  - Quality score trends
- ✅ Time range filters work
- ✅ Historical data visible

**Test File**: `tests/ui/test_agent_detail.py` (Playwright)

---

### Test Case 6.6: Trace Analyzer - AI Filtering
**Objective**: Verify AI-specific filtering in Trace Analyzer

**Test Steps**:
1. Navigate to Trace Analyzer
2. Apply "AI traces only" filter
3. Filter by agent attributes

**Expected Results**:
- ✅ "AI traces only" option available
- ✅ Filters by gen_ai.* attributes
- ✅ Only AI traces displayed
- ✅ Agent name filter works

**Test File**: `tests/ui/test_trace_analyzer_ai.py` (Playwright)

---

### Test Case 6.7: Trace View - AI Details Tab
**Objective**: Verify AI details tab in Trace View

**Test Steps**:
1. Open a trace with AI workflow
2. Click top-level workflow span
3. Navigate to "AI details" tab

**Expected Results**:
- ✅ "AI details" tab visible
- ✅ Metadata displayed:
  - Agent/Workflow name
  - Provider
  - Model
  - Framework
- ✅ Quality scores shown:
  - Bias
  - Toxicity
  - Hallucination
  - Relevance
  - Sentiment
- ✅ Agent input/output displayed
- ✅ Token usage visible

**Test File**: `tests/ui/test_trace_ai_details.py` (Playwright)

---

### Test Case 6.8: Agent Flow Visualization
**Objective**: Verify agent flow visualization in Trace View

**Test Steps**:
1. Open trace with multi-step agent
2. View agent flow visualization
3. Verify step representation

**Expected Results**:
- ✅ Agent flow diagram displayed
- ✅ Shows all agent steps
- ✅ Tool calls visible
- ✅ LLM calls highlighted
- ✅ Interactive navigation

**Test File**: `tests/ui/test_agent_flow.py` (Playwright)

---

### Test Case 6.9: Log Observer - AI Call Logs
**Objective**: Verify AI call logs in Log Observer

**Test Steps**:
1. Navigate to Log Observer
2. Filter for AI call logs
3. Verify log parsing and correlation

**Expected Results**:
- ✅ AI call logs parsed correctly
- ✅ Trace/span information present
- ✅ Navigation to related traces works
- ✅ Log fields extracted properly

**Test File**: `tests/ui/test_log_observer_ai.py` (Playwright)

---

## 7. Metrics and Dimensions Verification

### Test Case 7.1: Agent MMS Existence
**Objective**: Verify agent Monitoring MetricSet exists

**Test Steps**:
1. Navigate to Chart Builder
2. Search for "agent" MMS
3. Verify availability

**Expected Results**:
- ✅ agent MMS exists
- ✅ Accessible in Chart Builder
- ✅ Accessible in SignalFlow

**Test File**: `tests/ui/test_agent_mms.py` (Playwright)

---

### Test Case 7.2: Agent MMS Dimensions
**Objective**: Verify required dimensions for agent MMS

**Test Steps**:
1. Select agent MMS in Chart Builder
2. Check available dimensions
3. Verify each dimension works

**Expected Dimensions**:
- ✅ `sf_environment`
- ✅ `gen_ai.agent.name`
- ✅ `sf_error`
- ✅ `gen_ai.provider.name`
- ✅ `gen_ai.request.model`

**Test File**: `tests/ui/test_agent_dimensions.py` (Playwright)

---

### Test Case 7.3: Custom Dimensions
**Objective**: Verify custom dimensions can be added

**Test Steps**:
1. Add custom dimension to agent MMS
2. Verify it appears in charts
3. Test filtering by custom dimension

**Expected Results**:
- ✅ Custom dimensions addable
- ✅ Visible in Chart Builder
- ✅ Filtering works
- ✅ Aggregations work

**Test File**: `tests/ui/test_custom_dimensions.py` (Playwright)

---

### Test Case 7.4: Histogram Functions
**Objective**: Verify histogram functions on agent MMS

**Test Steps**:
1. Apply count() function
2. Apply min() function
3. Apply max() function
4. Apply median() function
5. Apply percentile() function

**Expected Results**:
- ✅ count() works correctly
- ✅ min() returns minimum value
- ✅ max() returns maximum value
- ✅ median() calculates correctly
- ✅ percentile(90) works
- ✅ All functions in Chart Builder
- ✅ All functions in SignalFlow

**Test File**: `tests/ui/test_histogram_functions.py` (Playwright)

---

## Test Execution Strategy

### Phase 1: Local Verification (Week 1)
1. Run all configuration tests locally
2. Verify telemetry generation with console exporters
3. Test all instrumentation methods
4. Document any issues

### Phase 2: lab0 Integration (Week 2)
1. Deploy to lab0 environment
2. Run all tests against lab0 tenant
3. Verify telemetry in Splunk APM
4. Test evaluation results

### Phase 3: UI Verification (Week 3)
1. Execute all Playwright UI tests
2. Verify Agents page functionality
3. Test navigation and filtering
4. Validate metrics and dimensions

### Phase 4: End-to-End Scenarios (Week 4)
1. Run complete user journeys
2. Test edge cases and error conditions
3. Performance and load testing
4. Final documentation

---

## Test Execution Commands

### Run All Tests
```bash
cd azure-ai-validation
pytest tests/ -v --html=logs/test_report.html
```

### Run Specific Category
```bash
# Configuration tests
pytest tests/test_*_config*.py -v

# UI tests
pytest tests/ui/ -v --headed

# Integration tests
pytest tests/test_*_integration*.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

---

## Test Reporting

### TestRail Integration
- Create test run for Alpha release
- Link test cases to requirements
- Update results after each execution
- Track defects and blockers

### Report Format
```
Test Case ID: TC-ALPHA-XXX
Status: PASS/FAIL/BLOCKED
Environment: lab0
Execution Date: YYYY-MM-DD
Tester: [Name]
Notes: [Observations]
Screenshots: [Links]
```

---

## Success Criteria

### Must Pass (P0)
- ✅ All instrumentation methods work
- ✅ Telemetry reaches Splunk APM
- ✅ Agents page displays correctly
- ✅ Trace View shows AI details
- ✅ Evaluation results visible

### Should Pass (P1)
- ✅ All configuration options work
- ✅ Filtering and sorting functional
- ✅ Navigation links work
- ✅ Metrics and dimensions available

### Nice to Have (P2)
- ✅ Performance optimized
- ✅ UI responsive
- ✅ Debug logging helpful
- ✅ Documentation accurate

---

## Contact and Support

**Test Lead**: [Your Name]  
**Environment**: lab0  
**Splunk Tenant**: [lab0 URL]  
**Documentation**: See `docs/` directory  
**Issues**: Track in JIRA/TestRail

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Ready for Execution
