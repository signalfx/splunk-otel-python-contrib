# Alpha Release Test Applications

## Overview

This directory contains production-ready test applications for validating Alpha release features. Each application is adapted from existing, well-tested examples and configured for comprehensive testing.

---

## 📱 Available Applications

### 1. **LangChain Evaluation App** (`langchain_evaluation_app.py`)

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

**Usage**:
```bash
# Basic run
python langgraph_travel_planner_app.py

# With poisoning
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

### 4. **Direct Azure OpenAI App** (`direct_azure_openai_app.py`)

**Purpose**: Direct Azure OpenAI usage without LangChain/LangGraph frameworks

**Features**:
- ✅ **LLMInvocation**: Direct LLM call instrumentation
- ✅ **AgentInvocation**: Agent-level instrumentation
- ✅ **No Framework**: Pure OpenAI client usage
- ✅ **Manual Instrumentation**: Explicit telemetry control

**Usage**:
```bash
# Test LLMInvocation
python direct_azure_openai_app.py --mode llm

# Test AgentInvocation
python direct_azure_openai_app.py --mode agent

# Test both
python direct_azure_openai_app.py --mode all
```

**Configuration**: `config/.env.lab0` (uses Azure OpenAI credentials)

**Validates**:
- ✅ LLMInvocation usage
- ✅ AgentInvocation usage
- ✅ Direct OpenAI client
- ✅ Manual span creation
- ✅ Token usage tracking
- ✅ Message content capture

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd alpha-release-testing

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

**Status**: Ready for Testing  
**Last Updated**: November 2025  
**Environment**: lab0 (Splunk Observability Cloud)
