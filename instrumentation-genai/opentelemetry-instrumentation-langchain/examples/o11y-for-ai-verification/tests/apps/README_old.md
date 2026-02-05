# Direct Azure OpenAI App V2 - Enhanced Evaluation Testing

**Status:** ✅ Complete - All P0, P1, P2 enhancements implemented

---

## 📋 Quick Reference

**File:** `direct_azure_openai_app_v2.py`  
**Type:** Manual GenAI instrumentation with multi-agent workflow  
**Scenarios:** 5 targeted evaluation tests (baseline, bias, toxicity, hallucination, relevance)  
**Features:** Config validation, retry logic, structured logging, metrics collection, cost tracking

---

## 🔄 Application Flow
- 🔄 **Intelligent Routing**: Cost-optimized, performance-optimized, reliability-optimized, round-robin strategies
- 💰 **Real-time Cost Tracking**: Per-request, per-provider, per-scenario cost analytics
- 🛡️ **Enterprise Features**: Automatic fallback, retry logic, circuit breakers, rate limiting
- 📊 **Advanced Observability**: OpenTelemetry integration, custom metrics, evaluation pipeline
- 🎯 **10 Test Scenarios**: Multi-agent retail, RAG pipeline, streaming TTFT, evaluation testing, and more

**Provider Architecture**:
```
LLMProviderFactory (Intelligent Routing)
├─ OpenAI Provider (gpt-4o-mini: $0.00015/1K tokens)
├─ Azure OpenAI Provider (gpt-4.1: $0.00030/1K tokens) [DEFAULT]
└─ Anthropic Provider (claude-3.5-sonnet: $0.003/1K tokens)
```

**Routing Strategies**:
1. **Cost-Optimized**: Automatically selects cheapest provider
2. **Performance-Optimized**: Selects fastest provider based on latency metrics
3. **Reliability-Optimized**: Selects provider with highest success rate
4. **Round-Robin**: Distributes load evenly across providers
5. **Manual**: User-specified provider with automatic fallback

**Test Scenarios**:
1. **Multi-Agent Retail** - Foundation tests with orchestrator pattern
2. **LangGraph Workflow** - Multi-agent coordination
3. **RAG Pipeline (Enhanced)** - Vector DB operations, retrieval quality, detailed chunking, 5 sample documents
4. **Streaming TTFT (Enhanced)** - Time-to-first-token metrics, P50/P95/P99, SLA validation, failure simulation
5. **Multi-Provider Edge Cases** - Provider switching, concurrent requests, error handling
6. **LiteLLM Proxy** - Provider attribution, metrics, trace correlation
7. **Evaluation Queue Management** (TC-PI2-INST-EVAL-01) - Queue rate-limiting, overflow handling
8. **Evaluation Error Handling** (TC-PI2-INST-EVAL-02) - Retry logic, graceful degradation
9. **Evaluation Monitoring Metrics** (TC-PI2-INST-EVAL-03) - Duration, token usage, queue size
10. **Retail Evaluation Tests** - Comprehensive evaluation metric testing with 6 test scenarios covering all 5 metrics (bias, toxicity, hallucination, relevance, sentiment)

**Usage Examples**:
```bash
# Use Azure OpenAI (default fallback)
python unified_genai_test_app.py --scenario eval_queue_management --provider azure

# Auto-routing with fallback
python unified_genai_test_app.py --scenario eval_error_handling --provider azure --fallback openai

# Cost-optimized routing (automatically selects cheapest)
python unified_genai_test_app.py --scenario streaming_ttft --routing-strategy cost_optimized

# Performance-optimized (automatically selects fastest)
python unified_genai_test_app.py --scenario rag_pipeline --routing-strategy performance_optimized

# Reliability-optimized (highest success rate)
python unified_genai_test_app.py --scenario litellm_proxy --routing-strategy reliability_optimized

# Round-robin load balancing
python unified_genai_test_app.py --scenario all --routing-strategy round_robin

# Run retail evaluation tests (all 6 scenarios)
python unified_genai_test_app.py --scenario retail_evaluation_tests

# Run all scenarios
python unified_genai_test_app.py --scenario all --provider azure
```

**Configuration**: `config/.env` (supports OpenAI, Azure OpenAI, Anthropic)

**Environment Variables**:
```bash
# Azure OpenAI (Default)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# OpenAI (Fallback)
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL_NAME=gpt-4o-mini

# Anthropic (Optional)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Cost Tracking Output**:
```
================================================================================
📊 LLM PROVIDER STATISTICS
================================================================================
Routing Strategy: cost_optimized
Total Requests: 150
Total Cost: $0.045000

Provider Breakdown:
--------------------------------------------------------------------------------

AZURE:
  Model: gpt-4.1
  Requests: 100
  Cost: $0.030000
  Avg Latency: 1245.67ms
  Success Rate: 99.00%
  Errors: 1

OPENAI:
  Model: gpt-4o-mini
  Requests: 50
  Cost: $0.015000
  Avg Latency: 987.34ms
  Success Rate: 100.00%
  Errors: 0
================================================================================
```

**Validates**:
- ✅ Multi-provider LLM support with seamless switching
- ✅ Intelligent routing strategies (cost, performance, reliability)
- ✅ Automatic fallback and retry logic
- ✅ Real-time cost tracking and analytics
- ✅ Provider performance monitoring (latency, success rate)
- ✅ Circuit breaker pattern for failing providers
- ✅ Enterprise-grade error handling
- ✅ Comprehensive evaluation pipeline (queue, errors, metrics)
- ✅ All 10 test scenarios with unified architecture
- ✅ Comprehensive evaluation testing with 6 retail scenarios

**Key Advantages**:
- **No Vendor Lock-in**: Switch providers instantly without code changes
- **Cost Optimization**: Automatically route to cheapest provider
- **High Availability**: Automatic fallback ensures zero downtime
- **Performance Insights**: Track which provider is fastest for your workload
- **Budget Control**: Real-time cost visibility per provider
- **Production-Ready**: Enterprise features like retry logic, circuit breakers

---

### 1. **Retail Shop LangChain App** (`retail_shop_langchain_app.py`) ⭐ NEW

**Purpose**: Multi-agent retail system with unified trace validation

**Features**:
- ✅ **3-Agent Hierarchy**: Store Manager (parent) → Inventory Agent + Customer Service Agent (children)
- ✅ **LangChain Auto-Instrumentation**: Uses `create_agent()` and `LangchainInstrumentor().instrument()`
- ✅ **Unified Traces**: Root span wrapper ensures single trace per scenario
- ✅ **Tool Functions**: `check_inventory()`, `get_return_policy()`, `format_response()`
- ✅ **Normal Content**: Demonstrates passing evaluation metrics

**Test Scenarios**:
1. **Product Availability** - Customer inquires about iPhone 15 Pro stock
2. **Return Request** - Customer requests laptop return process

**Usage**:
```bash
# Run both scenarios
python retail_shop_langchain_app.py

# Verify in Splunk APM
# Service: retail-shop-langchain
# Environment: From OTEL_DEPLOYMENT_ENVIRONMENT
```

**Configuration**: `config/.env`

**Validates**:
- ✅ LangChain automatic instrumentation
- ✅ Unified trace structure with root spans
- ✅ Multi-agent coordination
- ✅ Evaluation metrics on all agents
- ✅ Environment variable configuration
- ✅ Tool execution tracking

---

### 2. **LangChain Evaluation App** (`langchain_evaluation_app.py`)

**Source**: `qse-evaluation-harness/multi-agent-openai-metrics-trigger.py`

**Purpose**: Deterministic testing of evaluation metrics with LangChain multi-agent workflow

**Features**:
- ✅ **2-Agent Workflow**: Problematic Response Generator + Formatter
- ✅ **6 Test Scenarios**: Bias, Hallucination, Sentiment, Toxicity, Relevance, Comprehensive
- ✅ **Auto-Instrumentation**: Pure LangChain instrumentation
- ✅ **Evaluation Metrics**: All major metrics (bias, hallucination, sentiment, toxicity, relevance)
- ✅ **Deterministic**: Consistent, repeatable results

**Test Scenarios**:
1. **Bias Detection** - Tests biased content detection
2. **Hallucination Detection** - Tests factual accuracy validation
3. **Sentiment Analysis** - Tests sentiment classification
4. **Toxicity Detection** - Tests harmful content detection
5. **Relevance Assessment** - Tests context relevance
6. **Comprehensive Test** - Tests multiple metrics simultaneously

**Usage**:
```bash
# Run all scenarios
TEST_MODE=all python langchain_evaluation_app.py

# Run specific scenario
SCENARIO_INDEX=0 python langchain_evaluation_app.py  # Bias detection
SCENARIO_INDEX=1 python langchain_evaluation_app.py  # Hallucination detection

# With custom model
OPENAI_MODEL_NAME=gpt-4 SCENARIO_INDEX=2 python langchain_evaluation_app.py
```

**Configuration**: `config/.env.langchain`

**Validates**:
- ✅ LangChain instrumentation
- ✅ Multi-agent workflows
- ✅ Evaluation metrics generation
- ✅ Agent name configuration
- ✅ Token usage metrics
- ✅ Span hierarchy

---

### 2. **LangGraph Travel Planner App** (`langgraph_travel_planner_app.py`)

**Source**: `multi_agent_travel_planner/main.py`

**Purpose**: Multi-agent travel planning with LangGraph workflow orchestration

**Features**:
- ✅ **LangGraph StateGraph**: 5 specialized agents with conditional routing
- ✅ **Prompt Poisoning**: Configurable quality degradation for testing
- ✅ **Tool Usage**: Mock tools (flights, hotels, activities)
- ✅ **Workflow Orchestration**: State management, conditional edges
- ✅ **Comprehensive Telemetry**: Workflow, step, agent, and LLM spans

**Agents**:
1. **Coordinator** - Interprets traveler request, outlines plan
2. **Flight Specialist** - Selects flights (uses `mock_search_flights`)
3. **Hotel Specialist** - Recommends hotels (uses `mock_search_hotels`)
4. **Activity Specialist** - Curates activities (uses `mock_search_activities`)
5. **Plan Synthesizer** - Combines outputs into final itinerary

**Poisoning Configuration**:
```bash
# Probability of poisoning (0-1)
export TRAVEL_POISON_PROB=0.35

# Types of poisoning
export TRAVEL_POISON_TYPES=hallucination,bias,irrelevance,negative_sentiment,toxicity

# Maximum snippets per step
export TRAVEL_POISON_MAX=2

# Deterministic seed
export TRAVEL_POISON_SEED=42
```

**Instrumentation Modes**:

This app supports **BOTH zero-code and manual instrumentation** to meet customer documentation requirements (TC-1.1, TC-2.2, TC-2.3):

**🔵 Zero-Code Mode (Recommended for Production)**
```bash
opentelemetry-instrument python langgraph_travel_planner_app.py
```
**When to use**:
- ✅ Production deployments
- ✅ CI/CD pipelines  
- ✅ No code changes allowed
- ✅ Standard observability

**Pros**: No code changes, automatic patching, easier deployment  
**Cons**: Breaks IDE debuggers, less customization

**🟢 Manual Mode (Development/Debug)**
```bash
python langgraph_travel_planner_app.py
```
**When to use**:
- ✅ Development/debugging
- ✅ IDE breakpoints needed
- ✅ Custom instrumentation
- ✅ Advanced use cases

**Pros**: Full control, IDE debugging, custom spans  
**Cons**: Requires code changes, more maintenance

**Note**: Both modes generate identical telemetry. The app has manual instrumentation hardcoded, so zero-code mode adds a second layer (which is fine for testing comparison).

**Usage**:
```bash
# Zero-code mode (recommended)
opentelemetry-instrument python langgraph_travel_planner_app.py

# Manual mode
python langgraph_travel_planner_app.py

# With poisoning (both modes)
TRAVEL_POISON_PROB=0.75 TRAVEL_POISON_SEED=42 opentelemetry-instrument python langgraph_travel_planner_app.py
TRAVEL_POISON_PROB=0.75 TRAVEL_POISON_SEED=42 python langgraph_travel_planner_app.py

# Specific poison types
TRAVEL_POISON_TYPES=hallucination,bias python langgraph_travel_planner_app.py
```

**Configuration**: `config/.env.langgraph`

**Validates**:
- ✅ LangGraph workflow instrumentation
- ✅ Multi-agent coordination
- ✅ Tool execution spans
- ✅ Workflow name configuration
- ✅ Agent name configuration
- ✅ State management
- ✅ Conditional routing
- ✅ Quality degradation testing

---

### 3. **Traceloop Travel Planner App** (`traceloop_travel_planner_app.py`)

**Source**: `multi_agent_travel_planner/traceloop/main_traceloop.py`

**Purpose**: Demonstrate Traceloop SDK with automatic attribute translation

**Features**:
- ✅ **Traceloop SDK**: @workflow and @task decorators
- ✅ **Zero-Code Translator**: Automatic `traceloop.*` → `gen_ai.*` translation
- ✅ **Same Travel Logic**: Reuses travel planning workflow
- ✅ **Attribute Mapping**: Validates translator functionality

**Traceloop Decorators**:
```python
@workflow(name="travel_planning_workflow")
def plan_trip(request):
    # Workflow logic
    pass

@task(name="coordinator_task")
def coordinate(state):
    # Task logic
    pass
```

**Attribute Translation**:
- `traceloop.entity.name` → `gen_ai.agent.name`
- `traceloop.workflow.name` → `gen_ai.workflow.name`
- `traceloop.association.properties.*` → `gen_ai.*`

**Usage**:
```bash
# Basic run
python traceloop_travel_planner_app.py

# With DeepEval telemetry disabled
DEEPEVAL_TELEMETRY_OPT_OUT=YES python traceloop_travel_planner_app.py
```

**Configuration**: `config/.env.traceloop`

**Validates**:
- ✅ Traceloop SDK integration
- ✅ Translator installation
- ✅ Attribute translation (traceloop.* → gen_ai.*)
- ✅ DEEPEVAL_TELEMETRY_OPT_OUT
- ✅ Zero-code instrumentation

---

### 4. **Direct Azure OpenAI App** (`direct_azure_openai_app.py`) ⭐ ENHANCED

**Purpose**: Multi-department organizational workflow with manual GenAI instrumentation

**Features**:
- ✅ **4-Department Hierarchy**: Customer Service, Legal, Research, HR (all reporting to parent)
- ✅ **Manual GenAI Instrumentation**: Uses `LLMInvocation` and `AgentInvocation` directly
- ✅ **2 Scenarios**: Billing inquiry + Market analysis (both normal content)
- ✅ **Enhanced Telemetry**: 300s wait time for async evaluations, dual force flush
- ✅ **Azure OpenAI**: Direct Azure OpenAI client usage without frameworks

**Recent Enhancements (Nov 12)**:
- Increased telemetry wait time: 120s → 300s (matching langgraph app)
- Simplified scenarios to normal content for consistent evaluation metrics
- Added dual force flush mechanism for reliable telemetry export
- Verified all 5 evaluation metrics appear on all agents

**Architecture**:
```
Parent Agent (Organizational Coordinator)
├─ Customer Service Agent
├─ Legal Compliance Agent
├─ Research Analysis Agent
└─ HR Agent
```

**Usage**:
```bash
# Run both scenarios
python direct_azure_openai_app.py

# Verify in Splunk APM
# Service: direct-azure-openai-test
# Environment: From OTEL_DEPLOYMENT_ENVIRONMENT
```

**Configuration**: `config/.env` (uses Azure OpenAI credentials)

**Validates**:
- ✅ Manual GenAI instrumentation (LLMInvocation, AgentInvocation)
- ✅ Multi-agent hierarchical workflows
- ✅ Direct Azure OpenAI client usage
- ✅ Manual span creation and management
- ✅ Token usage tracking
- ✅ Message content capture
- ✅ Evaluation metrics on all agents
- ✅ Async evaluation completion with proper wait times

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd o11y-for-ai-verification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
# Switch to lab0 realm
./scripts/switch_realm.sh lab0

# Or manually configure
cp config/.env.lab0.template config/.env
vim config/.env  # Add your credentials
```

### 3. Run Test Applications

```bash
cd tests/apps

# LangChain evaluation
python langchain_evaluation_app.py

# LangGraph travel planner
python langgraph_travel_planner_app.py

# Traceloop travel planner
python traceloop_travel_planner_app.py

# Direct Azure OpenAI
python direct_azure_openai_app.py
```

---

## 🐳 Docker Deployment

### Build Image
```bash
cd o11y-for-ai-verification
docker build -t alpha-test-apps:latest .
```

### Run Individual Apps

#### LangChain Evaluation (Zero-Code)
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  alpha-test-apps:latest \
  opentelemetry-instrument python tests/apps/langchain_evaluation_app.py
```

#### LangGraph Travel Planner (Zero-Code)
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -e TRAVEL_POISON_PROB=0.75 \
  alpha-test-apps:latest \
  opentelemetry-instrument python tests/apps/langgraph_travel_planner_app.py
```

#### LangGraph Travel Planner (Manual)
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  alpha-test-apps:latest \
  python tests/apps/langgraph_travel_planner_app.py
```

#### Traceloop Travel Planner
```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -e DEEPEVAL_TELEMETRY_OPT_OUT=YES \
  alpha-test-apps:latest \
  python tests/apps/traceloop_travel_planner_app.py
```

### Kubernetes CronJob Example

Create `k8s-alpha-test.yaml`:
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: alpha-test-langgraph-zerocode
spec:
  schedule: "*/30 * * * *"  # Every 30 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: test-runner
            image: alpha-test-apps:latest
            command: ["opentelemetry-instrument"]
            args: ["python", "tests/apps/langgraph_travel_planner_app.py"]
            env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://otel-collector:4317"
            - name: OTEL_RESOURCE_ATTRIBUTES
              value: "deployment.environment=alpha-test,flavor=zerocode"
            - name: OTEL_SERVICE_NAME
              value: "alpha-test-langgraph"
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: alpha-test-langgraph-manual
spec:
  schedule: "*/30 * * * *"  # Every 30 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: test-runner
            image: alpha-test-apps:latest
            args: ["python", "tests/apps/langgraph_travel_planner_app.py"]
            env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://otel-collector:4317"
            - name: OTEL_RESOURCE_ATTRIBUTES
              value: "deployment.environment=alpha-test,flavor=manual"
            - name: OTEL_SERVICE_NAME
              value: "alpha-test-langgraph"
          restartPolicy: OnFailure
```

Deploy:
```bash
kubectl apply -f k8s-alpha-test.yaml

# Check status
kubectl get cronjobs
kubectl get jobs
kubectl logs -l job-name=alpha-test-langgraph-zerocode-xxxxx
```

---

## 📊 Telemetry Generated

### LangChain Evaluation App
```
Spans:
- Agent 1 (Problematic Response Generator)
- Agent 2 (Response Formatter)
- OpenAI chat calls

Metrics:
- gen_ai.evaluation.bias
- gen_ai.evaluation.hallucination
- gen_ai.evaluation.sentiment
- gen_ai.evaluation.toxicity
- gen_ai.evaluation.relevance
- gen_ai.client.token.usage
- gen_ai.agent.duration
```

### LangGraph Travel Planner App
```
Spans:
- gen_ai.workflow LangGraph
- gen_ai.step (coordinator, flight_specialist, hotel_specialist, etc.)
- invoke_agent (for each agent)
- chat ChatOpenAI (LLM calls)
- tool (mock_search_flights, mock_search_hotels, etc.)

Metrics:
- gen_ai.workflow.duration
- gen_ai.agent.duration
- gen_ai.client.operation.duration
- gen_ai.client.token.usage
- gen_ai.evaluation.* (all evaluation metrics)

Attributes:
- gen_ai.workflow.name
- gen_ai.agent.name
- gen_ai.provider.name
- gen_ai.request.model
- travel.plan.poison_events (if poisoning enabled)
```

### Traceloop Travel Planner App
```
Spans:
- Workflow spans (with traceloop.workflow.name)
- Task spans (with traceloop.entity.name)
- Translated to gen_ai.* attributes

Attributes (after translation):
- gen_ai.workflow.name (from traceloop.workflow.name)
- gen_ai.agent.name (from traceloop.entity.name)
- gen_ai.* (from traceloop.association.properties.*)
```

### Direct Azure OpenAI App
```
Spans:
- LLMInvocation spans
- AgentInvocation spans
- Custom application spans

Metrics:
- gen_ai.client.token.usage
- gen_ai.client.operation.duration

Attributes:
- gen_ai.request.model
- gen_ai.provider.name
- gen_ai.framework
- gen_ai.operation.name
```

---

## 🧪 Testing Use Cases

### Use Case 1: Zero-Code vs Code-Based Instrumentation
```bash
# Zero-code (via opentelemetry-instrument)
opentelemetry-instrument python langchain_evaluation_app.py

# Code-based (instrumentation in code)
python langchain_evaluation_app.py
```

### Use Case 2: Agent Name Configuration
```bash
# LangChain - agent names set in code
python langchain_evaluation_app.py

# LangGraph - agent names in workflow
python langgraph_travel_planner_app.py

# Verify gen_ai.agent.name in spans
```

### Use Case 3: Workflow Name Configuration
```bash
# LangGraph - workflow name set
python langgraph_travel_planner_app.py

# Verify gen_ai.workflow.name in spans
```

### Use Case 4: Evaluation Metrics
```bash
# All evaluation metrics
python langchain_evaluation_app.py

# With poisoning for quality degradation
TRAVEL_POISON_PROB=0.75 python langgraph_travel_planner_app.py
```

### Use Case 5: Traceloop Translator
```bash
# Run Traceloop app
python traceloop_travel_planner_app.py

# Verify attribute translation in spans
# traceloop.* → gen_ai.*
```

### Use Case 6: Direct AI Instrumentation
```bash
# LLMInvocation
python direct_azure_openai_app.py --mode llm

# AgentInvocation
python direct_azure_openai_app.py --mode agent
```

---

## 🔍 Verification

### Check Telemetry in Splunk APM

1. **Navigate to Splunk APM** (lab0 tenant)
2. **Go to Agents Page**
   - Verify agents appear
   - Check agent names
   - View metrics (requests, errors, latency, tokens)

3. **Open Trace View**
   - Find traces from test apps
   - Verify span hierarchy
   - Check AI details tab
   - View evaluation scores

4. **Check Metrics**
   - Navigate to Metrics Explorer
   - Search for `gen_ai.*` metrics
   - Verify agent MMS
   - Check dimensions

---

## 📝 Configuration Files

### `.env.langchain` (LangChain Evaluation App)
```bash
OPENAI_API_KEY=your-key
OPENAI_MODEL_NAME=gpt-4o-mini
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=langchain-evaluation-test
```

### `.env.langgraph` (LangGraph Travel Planner)
```bash
OPENAI_API_KEY=your-key
TRAVEL_POISON_PROB=0.35
TRAVEL_POISON_TYPES=hallucination,bias,irrelevance,negative_sentiment,toxicity
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=langgraph-travel-planner-test
```

### `.env.traceloop` (Traceloop Travel Planner)
```bash
OPENAI_API_KEY=your-key
DEEPEVAL_TELEMETRY_OPT_OUT=YES
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=traceloop-travel-planner-test
```

---

## 🔧 Complete Environment Variables Reference

### Required Variables
| Variable | Purpose | Example | Notes |
|----------|---------|---------|-------|
| `OPENAI_API_KEY` | OpenAI authentication | `sk-proj-...` | Required for all apps |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | `http://localhost:4317` | gRPC protocol |
| `OTEL_SERVICE_NAME` | Service identifier | `alpha-release-test` | Appears in APM |

### Optional Core Configuration
| Variable | Purpose | Default | Apps |
|----------|---------|---------|------|
| `OPENAI_MODEL_NAME` | Model selection | `gpt-4o-mini` | All |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture prompts/responses | `true` | All |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitter types | `span_metric_event,splunk` | All |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | Content capture mode | `SPAN_AND_EVENT` | All |
| `OTEL_RESOURCE_ATTRIBUTES` | Resource attributes | `deployment.environment=alpha` | All |
| `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE` | Metrics temporality | `DELTA` | All |

### LangGraph Poisoning (Optional)
| Variable | Purpose | Default | Range/Values |
|----------|---------|---------|-------------|
| `TRAVEL_POISON_PROB` | Poisoning probability | `0.8` | `0.0-1.0` |
| `TRAVEL_POISON_TYPES` | Poison types to inject | `hallucination,bias,irrelevance,negative_sentiment,toxicity` | CSV list |
| `TRAVEL_POISON_MAX` | Max snippets per step | `2` | `1-5` |
| `TRAVEL_POISON_SEED` | Deterministic seed | (random) | Any integer |

### Traceloop Specific
| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `DEEPEVAL_TELEMETRY_OPT_OUT` | Disable DeepEval telemetry | `NO` | Set to `YES` for Traceloop |
| `TRACELOOP_BASE_URL` | Traceloop API endpoint | - | Optional |

### Evaluation Configuration (Optional)
| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | Evaluators to use | `(Bias,Toxicity,Hallucination,Relevance,Sentiment)` | Tuple format |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION` | Aggregate results | `true` | Boolean |
| `OTEL_GENAI_EVAL_DEBUG_SKIPS` | Debug skipped evaluations | `false` | Boolean |
| `OTEL_GENAI_EVAL_DEBUG_EACH` | Debug each evaluation | `false` | Boolean |

---

## 📦 Dependencies & Requirements

### Core Requirements
```txt
# OpenTelemetry Core
opentelemetry-sdk>=1.38.0
opentelemetry-api>=1.38.0
opentelemetry-instrumentation>=0.48b0

# OpenTelemetry Exporters
opentelemetry-exporter-otlp>=1.38.0
opentelemetry-exporter-otlp-proto-grpc>=1.38.0

# LangChain/LangGraph
langchain>=1.0.0
langchain-openai>=1.0.0
langchain-core>=1.0.0
langgraph>=1.0.0

# OpenAI
openai>=1.0.0
```

### Splunk Packages (Install from local)
```bash
# Install in this order
pip install -e ../../../../util/opentelemetry-util-genai --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-emitters-splunk --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-evals --no-deps
pip install -e ../../../../util/opentelemetry-util-genai-evals-deepeval
pip install -e ../../../../instrumentation-genai/opentelemetry-instrumentation-langchain/
```

### Evaluation Requirements
```txt
deepeval>=0.21.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### Traceloop Requirements (Separate venv recommended)
```txt
traceloop-sdk>=0.47.4
```

### ⚠️ Dependency Conflicts

**DeepEval vs Traceloop**: These packages have conflicting dependencies. Solutions:

1. **Separate Virtual Environments** (Recommended):
   ```bash
   # For LangChain/LangGraph apps
   python -m venv .venv-langchain
   source .venv-langchain/bin/activate
   pip install -r requirements-langchain.txt
   
   # For Traceloop app
   python -m venv .venv-traceloop
   source .venv-traceloop/bin/activate
   pip install -r requirements-traceloop.txt
   ```

2. **Use run_tests.sh**: The automated test runner handles environment switching automatically.

### Minimum Python Version
- **Python 3.8+** required
- **Python 3.10+** recommended for best compatibility

---

## 🐛 Troubleshooting

### Issue: OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Test connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Issue: No Telemetry
```bash
# Check OTEL Collector
curl http://localhost:4317

# Use console exporter for debugging
export OTEL_TRACES_EXPORTER=console
python langchain_evaluation_app.py
```

### Issue: Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check installations
pip list | grep -E "langchain|opentelemetry|traceloop"
```

---

## 📚 Documentation

- **Test Plan**: `../docs/ALPHA_RELEASE_TEST_PLAN.md`
- **Implementation Plan**: `../IMPLEMENTATION_PLAN.md`
- **Resource Analysis**: `../RESOURCE_ANALYSIS.md`
- **Configuration Guide**: `../config/README.md`

---

## 📊 Application Comparison Matrix

| Feature | Retail Shop | LangChain Eval | LangGraph Travel | Direct Azure | Traceloop |
|---------|-------------|----------------|------------------|--------------|-----------|
| **Instrumentation** | LangChain Auto | LangChain Auto | LangGraph | Manual GenAI | Traceloop SDK |
| **Agent Count** | 3 (1+2) | 2 | 5 | 5 (1+4) | 5 |
| **Scenarios** | 2 | 6 | 1 | 2 | 1 |
| **Unified Traces** | ✅ Root span | ❌ Separate | ✅ Workflow | ✅ Parent span | ✅ Workflow |
| **Tool Usage** | ✅ 3 tools | ❌ No tools | ✅ Mock tools | ❌ No tools | ✅ Mock tools |
| **Content Type** | Normal | Problematic | Normal/Poisoned | Normal | Normal |
| **Eval Metrics** | ✅ All 5 | ✅ All 5 | ✅ All 5 | ✅ All 5 | ✅ All 5 |
| **Use Case** | Unified traces | Metric testing | Workflow orchestration | Manual instrumentation | SDK translation |
| **Status** | ⭐ NEW | Reference | Existing | ⭐ ENHANCED | Existing |

---

## ✅ Success Criteria

Each application should:
- ✅ Run without errors
- ✅ Generate telemetry (spans, metrics, logs)
- ✅ Export to OTLP endpoint
- ✅ Appear in Splunk APM
- ✅ Show correct agent/workflow names
- ✅ Generate evaluation metrics
- ✅ Complete within reasonable time (<5 minutes)

---

## 🎯 Key Takeaways

### **For Unified Traces**
Use **Retail Shop App** or **Direct Azure App** - both demonstrate root span patterns for single trace per workflow.

### **For Evaluation Metrics Testing**
Use **LangChain Eval App** - 6 scenarios specifically designed to trigger different evaluation metrics.

### **For Workflow Orchestration**
Use **LangGraph Travel App** - demonstrates complex state management and conditional routing.

### **For Manual Instrumentation**
Use **Direct Azure App** - shows how to use `LLMInvocation` and `AgentInvocation` directly without frameworks.

### **For SDK Integration**
Use **Traceloop App** - validates attribute translation from Traceloop SDK to GenAI conventions.

---

**Status**: Ready for Testing  
**Last Updated**: November 12, 2025  
**Environment**: RC0 (ai-test-val) & Lab0 (Splunk Observability Cloud)
