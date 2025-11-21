# Direct Azure OpenAI Application - Multi-Department Organization Workflow

## Overview

This application demonstrates a **realistic multi-department organization** with hierarchical agent communication and different evaluation patterns for each agent type. It tests GenAI instrumentation without any AI framework (no LangChain, no LangGraph) using direct Azure OpenAI SDK calls.

## Organization Structure

```
┌─────────────────────────────────────────┐
│   Research Department (Parent Agent)   │
│   Evals: Relevance, Hallucination      │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┼─────────┬─────────────┐
        │         │         │             │
   ┌────▼───┐ ┌──▼────┐ ┌─▼──────┐ ┌────▼─────┐
   │Customer│ │ Legal │ │Research│ │    HR    │
   │Service │ │  &    │ │Analysis│ │          │
   │        │ │Compli-│ │        │ │          │
   │        │ │ ance  │ │        │ │          │
   └────┬───┘ └───┬───┘ └───┬────┘ └────┬─────┘
        │         │         │            │
   ┌────▼───┐ ┌──▼────┐ ┌──▼─────┐ ┌───▼──────┐
   │Support │ │Contract│ │Market  │ │Recruiting│
   │Tier-1  │ │Review │ │Intel   │ │          │
   └────────┘ └────────┘ └────────┘ └──────────┘
```

## Agent Hierarchy

### Level 1: Parent Agent
- **Research Department Coordinator**
  - **Role**: Orchestrates all departments
  - **Evaluation**: Relevance, Hallucination
  - **Responsibilities**: Route requests, synthesize responses

### Level 2: Department Agents

| Department | Agent Type | Evaluation Metrics | Purpose |
|------------|------------|-------------------|---------|
| **Customer Service** | `customer_support` | Toxicity, Sentiment | Customer-facing communication |
| **Legal & Compliance** | `legal_review` | Bias, Hallucination | Accuracy-critical legal review |
| **Research & Analysis** | `research` | Relevance, Hallucination | Information quality |
| **Human Resources** | `human_resources` | Bias, Toxicity, Sentiment | Fairness-critical HR decisions |

### Level 3: Sub-Department Agents

| Sub-Department | Parent | Agent Type | Focus |
|----------------|--------|------------|-------|
| **Support Tier-1** | Customer Service | `frontline_support` | First-line customer support |
| **Contract Review** | Legal & Compliance | `legal_analysis` | Contract analysis and risk assessment |
| **Market Intelligence** | Research & Analysis | `market_research` | Market trends and competitive analysis |
| **Recruiting** | Human Resources | `talent_acquisition` | Candidate evaluation |

## Test Scenarios

### Scenario 1: Customer Complaint Handling
**Evaluation Focus**: Toxicity, Sentiment

**Request**:
```
A customer is frustrated because their order was delayed by 2 weeks.
They want a refund and are threatening to leave negative reviews.
How should we respond?
```

**Expected Behavior**:
- ✅ Non-toxic responses
- ✅ Empathetic communication
- ✅ Sentiment analysis shows positive/neutral tone
- ✅ Customer Service → Support Tier-1 delegation

**Agents Involved**:
1. Research Dept Coordinator (Parent)
2. Customer Service Dept
3. Support Tier-1 (Sub-dept)
4. Legal, Research, HR (parallel consultation)

---

### Scenario 2: Legal Contract Review
**Evaluation Focus**: Bias, Hallucination

**Request**:
```
Review a vendor contract with the following terms:
- 3-year commitment with auto-renewal
- Liability cap at $50,000
- Data ownership remains with vendor
- 90-day termination notice required
What are the risks?
```

**Expected Behavior**:
- ✅ Unbiased legal analysis
- ✅ Factually accurate (no hallucinated clauses)
- ✅ Bias score near 0
- ✅ Hallucination score near 0

**Agents Involved**:
1. Research Dept Coordinator (Parent)
2. Legal & Compliance Dept
3. Contract Review (Sub-dept)
4. Customer Service, Research, HR (parallel consultation)

---

### Scenario 3: Market Intelligence Request
**Evaluation Focus**: Relevance, Hallucination

**Request**:
```
Analyze the competitive landscape for AI observability tools.
What are the key market trends and who are the main competitors?
```

**Expected Behavior**:
- ✅ Relevant market insights
- ✅ No fabricated data or companies
- ✅ High relevance score
- ✅ Low hallucination score

**Agents Involved**:
1. Research Dept Coordinator (Parent)
2. Research & Analysis Dept
3. Market Intelligence (Sub-dept)
4. Customer Service, Legal, HR (parallel consultation)

---

### Scenario 4: Candidate Evaluation
**Evaluation Focus**: Bias, Toxicity, Sentiment

**Request**:
```
Evaluate a candidate for Senior Engineer position:
- 8 years experience
- Strong technical skills
- Career gap of 2 years (personal reasons)
- Excellent interview performance
Should we proceed with an offer?
```

**Expected Behavior**:
- ✅ Fair, unbiased evaluation
- ✅ No discrimination based on career gap
- ✅ Respectful language
- ✅ Bias score near 0
- ✅ Toxicity score near 0

**Agents Involved**:
1. Research Dept Coordinator (Parent)
2. Human Resources Dept
3. Recruiting (Sub-dept)
4. Customer Service, Legal, Research (parallel consultation)

## Evaluation Patterns by Agent Type

### Customer Service Agents
**Metrics**: Toxicity, Sentiment
- **Why**: Customer-facing communication must be empathetic and non-toxic
- **Threshold**: Toxicity < 0.3, Sentiment > 0.5

### Legal & Compliance Agents
**Metrics**: Bias, Hallucination
- **Why**: Legal advice must be unbiased and factually accurate
- **Threshold**: Bias < 0.2, Hallucination < 0.1

### Research & Analysis Agents
**Metrics**: Relevance, Hallucination
- **Why**: Research must be relevant and based on real data
- **Threshold**: Relevance > 0.7, Hallucination < 0.2

### Human Resources Agents
**Metrics**: Bias, Toxicity, Sentiment
- **Why**: HR decisions must be fair, respectful, and unbiased
- **Threshold**: Bias < 0.1, Toxicity < 0.2, Sentiment > 0.6

## Telemetry & Instrumentation

### Span Hierarchy
```
research-dept-coordinator (Parent Agent)
├─ LLM Call: Routing Analysis
├─ customer-service-dept (Department Agent)
│  ├─ support-tier1 (Sub-department Agent)
│  │  └─ LLM Call: Support Response
│  └─ LLM Call: Department Synthesis
├─ legal-compliance-dept (Department Agent)
│  ├─ contract-review (Sub-department Agent)
│  │  └─ LLM Call: Contract Analysis
│  └─ LLM Call: Legal Opinion
├─ research-analysis-dept (Department Agent)
│  ├─ market-intelligence (Sub-department Agent)
│  │  └─ LLM Call: Market Research
│  └─ LLM Call: Research Summary
├─ hr-dept (Department Agent)
│  ├─ recruiting (Sub-department Agent)
│  │  └─ LLM Call: Candidate Evaluation
│  └─ LLM Call: HR Policy
└─ LLM Call: Final Synthesis
```

### GenAI Attributes
Each span includes:
- `gen_ai.request.model` (e.g., `gpt-4.1`)
- `gen_ai.provider.name` (`azure` or `openai`)
- `gen_ai.operation.name` (`chat.completions` or `invoke_agent`)
- `gen_ai.agent.name` (e.g., `customer-service-dept`)
- `gen_ai.agent.type` (e.g., `customer_support`)
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`

### Evaluation Metrics
Each agent type generates different evaluation metrics:
- `gen_ai.evaluation.toxicity`
- `gen_ai.evaluation.sentiment`
- `gen_ai.evaluation.bias`
- `gen_ai.evaluation.hallucination`
- `gen_ai.evaluation.relevance`

## Running the Application

### Prerequisites
```bash
# Ensure environment variables are set in config/.env
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_DEPLOYMENT=<your-deployment>
OTEL_SERVICE_NAME=direct-ai-app
OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment))"
```

### Execute
```bash
cd tests/apps
python direct_azure_openai_app.py
```

### Expected Output
```
🏢 MULTI-DEPARTMENT ORGANIZATION WORKFLOW
================================================================================
Testing hierarchical agent communication with evaluation patterns
================================================================================

Organization Structure:
  Parent: Research Department (Relevance, Hallucination)
  ├─ Customer Service (Toxicity, Sentiment)
  │  └─ Support Tier-1
  ├─ Legal & Compliance (Bias, Hallucination)
  │  └─ Contract Review
  ├─ Research & Analysis (Relevance, Hallucination)
  │  └─ Market Intelligence
  └─ Human Resources (Bias, Toxicity, Sentiment)
     └─ Recruiting
================================================================================

📋 SCENARIO 1: Customer Complaint Handling
================================================================================
Evaluation Focus: Toxicity, Sentiment (customer-facing)
Expected: Non-toxic, empathetic responses
================================================================================

🏢 RESEARCH DEPARTMENT (Parent Agent)
================================================================================
Request: A customer is frustrated...
    💬 LLM Call from Research Coordinator

  📞 Customer Service Department
    💬 LLM Call from Support Tier-1
    💬 LLM Call from Customer Service Manager
    ✓ Customer Service: Response prepared

  ⚖️  Legal & Compliance Department
    💬 LLM Call from Contract Review
    💬 LLM Call from Chief Legal Officer
    ✓ Legal & Compliance: Opinion issued

  🔬 Research & Analysis Department
    💬 LLM Call from Market Intelligence
    💬 LLM Call from Research Director
    ✓ Research & Analysis: Report completed

  👥 Human Resources Department
    💬 LLM Call from Recruiting
    💬 LLM Call from HR Director
    ✓ Human Resources: Guidance provided

    💬 LLM Call from Research Coordinator (Final Synthesis)

================================================================================
✅ ORGANIZATIONAL RESPONSE COMPLETE
================================================================================
🔍 Trace ID: a1b2c3d4e5f6...

✅ Scenario 1 Complete - Trace ID: a1b2c3d4e5f6...

[... 3 more scenarios ...]

================================================================================
✅ ALL SCENARIOS COMPLETE
================================================================================
Total Scenarios: 4
Total Departments: 4 (Customer Service, Legal, Research, HR)
Total Sub-departments: 4 (Support, Contract Review, Market Intel, Recruiting)
Total Agents: 9 (1 Parent + 4 Dept + 4 Sub-dept)
Total LLM Calls: ~27 (3 per sub-dept × 4 depts × 4 scenarios)

Trace IDs:
  Scenario 1 (Customer): a1b2c3d4e5f6...
  Scenario 2 (Legal):    b2c3d4e5f6a7...
  Scenario 3 (Research): c3d4e5f6a7b8...
  Scenario 4 (HR):       d4e5f6a7b8c9...

Evaluation Patterns Tested:
  ✓ Toxicity (Customer Service, HR)
  ✓ Sentiment (Customer Service, HR)
  ✓ Bias (Legal & Compliance, HR)
  ✓ Hallucination (Legal & Compliance, Research)
  ✓ Relevance (Research)
```

## Validation in Splunk APM

### Search Query
```
sf_service:direct-ai-app
```

### Filter by Scenario
```
sf_service:direct-ai-app AND trace.id:a1b2c3d4e5f6...
```

### Verification Checklist

#### Span Hierarchy
- [ ] Parent span: `research-dept-coordinator`
- [ ] 4 department spans (customer-service, legal-compliance, research-analysis, hr)
- [ ] 4 sub-department spans (support-tier1, contract-review, market-intelligence, recruiting)
- [ ] ~27 LLM invocation spans total

#### GenAI Attributes
- [ ] `gen_ai.request.model` = `gpt-4.1` (Azure) or `gpt-4o-mini` (OpenAI)
- [ ] `gen_ai.provider.name` = `azure` or `openai`
- [ ] `gen_ai.operation.name` = `chat.completions` or `invoke_agent`
- [ ] `gen_ai.agent.name` matches agent names
- [ ] `gen_ai.agent.type` matches agent types

#### Evaluation Metrics by Agent Type
- [ ] **Customer Service spans**: `gen_ai.evaluation.toxicity`, `gen_ai.evaluation.sentiment`
- [ ] **Legal spans**: `gen_ai.evaluation.bias`, `gen_ai.evaluation.hallucination`
- [ ] **Research spans**: `gen_ai.evaluation.relevance`, `gen_ai.evaluation.hallucination`
- [ ] **HR spans**: `gen_ai.evaluation.bias`, `gen_ai.evaluation.toxicity`, `gen_ai.evaluation.sentiment`

#### AI Details Section
- [ ] Model name displayed
- [ ] Provider displayed
- [ ] Token usage (input/output) displayed
- [ ] Message content captured

## Key Differences from Other Apps

| Feature | `direct_azure_openai_app.py` | `langchain_evaluation_app.py` | `langgraph_agent_example.py` |
|---------|------------------------------|-------------------------------|------------------------------|
| **Framework** | None (raw SDK) | LangChain | LangGraph |
| **Instrumentation** | Manual (TelemetryHandler) | Automatic | Automatic |
| **Agent Hierarchy** | 3 levels (Parent → Dept → Sub-dept) | 1 level (chain) | 2 levels (graph nodes) |
| **Evaluation Patterns** | Different per agent type | Uniform | Uniform |
| **Scenarios** | 4 realistic business scenarios | 1 RAG scenario | 1 travel planning scenario |
| **Complexity** | High (9 agents, 27 LLM calls) | Medium (1 chain) | High (5 agents, graph) |

## Test Coverage

### TC-3.2: Instrument a Python AI application
✅ **PASSED** - Direct Azure OpenAI SDK instrumented with `TelemetryHandler`

### TC-3.3: Configure instrumentation and evaluation settings
✅ **PASSED** - Different evaluation patterns per agent type via `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS`

## Troubleshooting

### No traces in Splunk APM
- Check `OTEL_EXPORTER_OTLP_ENDPOINT` is set to `http://localhost:4317`
- Verify OTEL collector is running
- Check trace IDs are printed in console output

### Evaluation metrics missing
- Ensure `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` is uncommented in `.env`
- Verify DeepEval is installed: `pip install deepeval`
- Check `OPENAI_API_KEY` is set (DeepEval uses OpenAI for evaluations)

### 401 Unauthorized error
- Verify `AZURE_OPENAI_API_KEY` is correct
- Check `AZURE_OPENAI_ENDPOINT` matches your Azure resource
- Ensure `AZURE_OPENAI_DEPLOYMENT` matches your deployment name

### Agent hierarchy not visible
- Filter by trace ID in Splunk APM
- Check span names match agent names
- Verify `gen_ai.agent.name` and `gen_ai.agent.type` attributes

## Architecture Highlights

### Why This Design?

1. **Realistic Business Scenario**: Models actual enterprise organization structure
2. **Different Evaluation Needs**: Different departments have different quality requirements
3. **Hierarchical Communication**: Tests parent-child agent relationships
4. **Manual Instrumentation**: Proves GenAI utilities work without frameworks
5. **Azure OpenAI Focus**: Tests Azure-specific authentication and configuration

### Evaluation Pattern Rationale

| Agent Type | Evaluation Metrics | Rationale |
|------------|-------------------|-----------|
| Customer Service | Toxicity, Sentiment | Customer satisfaction depends on empathetic, non-toxic communication |
| Legal & Compliance | Bias, Hallucination | Legal advice must be unbiased and factually accurate to avoid liability |
| Research & Analysis | Relevance, Hallucination | Research quality depends on relevant, fact-based insights |
| Human Resources | Bias, Toxicity, Sentiment | HR decisions must be fair, respectful, and legally compliant |

## Future Enhancements

- [ ] Add workflow-level evaluation (cross-department consistency)
- [ ] Implement conditional routing (skip departments based on request type)
- [ ] Add error handling and retry logic
- [ ] Implement caching for repeated queries
- [ ] Add performance metrics (latency, throughput)
- [ ] Support multiple LLM providers per department
