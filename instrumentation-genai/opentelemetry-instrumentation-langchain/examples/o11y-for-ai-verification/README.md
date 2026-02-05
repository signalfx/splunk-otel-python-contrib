# O11y AI Test Framework

**Automated testing for AI Observability with LangChain, LangGraph, and OpenTelemetry**

## Quick Start

```bash
# 1. Setup virtual environment (one-time)
./scripts/setup_venv.sh

# 2. Activate virtual environment (required)
source .venv/bin/activate

# 3. Configure Azure OpenAI
cp config/azure_openai.env.template config/azure_openai.env
# Edit config/azure_openai.env with your credentials

# 4. Run all tests
./scripts/run_all.sh
```

**Notes:** 
- `setup_venv.sh` creates the venv but doesn't activate it. You must run `source .venv/bin/activate` manually.
- Evaluation packages (`opentelemetry-util-genai-evals*`) are installed from local editable sources in `../../util/` for development.

**Time:** ~25-30 minutes for complete automation

---

## Prerequisites

- Python 3.10+
- Azure OpenAI credentials (primary LLM provider)
- Splunk Observability Cloud access (RC0 realm)

---

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_venv.sh` | Setup virtual environment & dependencies | `./scripts/setup_venv.sh` |
| `setup_environment.sh` | Configure environment variables | `source scripts/setup_environment.sh` |
| `start_collector.sh` | Start OpenTelemetry Collector | `./scripts/start_collector.sh` |
| `stop_collector.sh` | Stop OpenTelemetry Collector | `./scripts/stop_collector.sh` |
| `run_apps.sh` | Execute test applications | `./scripts/run_apps.sh [scenario] [iterations]` |
| `run_tests.sh` | Run test suites | `./scripts/run_tests.sh` |
| `run_all.sh` | **Complete automation** | `./scripts/run_all.sh` |

---

## Configuration

### Azure OpenAI (Required)

**Option 1: Config file (recommended)**
```bash
cp config/azure_openai.env.template config/azure_openai.env
# Edit with your credentials
```

**Option 2: Environment variables**
```bash
export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'
export AZURE_OPENAI_API_KEY='your-api-key'
export AZURE_OPENAI_DEPLOYMENT='gpt-4'
```

Configuration priority: `config/azure_openai.env` → `config/.env.rc0.template` → environment variables

---

## Test Scenarios

1. **multi_agent_retail** - Multi-agent retail assistant
2. **langgraph_workflow** - LangGraph state machine
3. **rag_pipeline** - RAG with vector DB
4. **streaming_ttft** - Streaming responses
5. **multi_provider_edge_cases** - Edge case handling
6. **litellm_proxy** - LiteLLM proxy integration
7. **eval_queue_management** - Evaluation queue
8. **eval_error_handling** - Evaluation errors
9. **eval_monitoring_metrics** - Evaluation metrics
10. **retail_evaluation_tests** - Retail evaluation

---

## Troubleshooting

### Tests fail with "Trace not available"

**Cause:** Test apps failed to run, no traces generated.

**Solution:**
```bash
# Verify dependencies installed
.venv/bin/python3 -c "from langchain_openai import ChatOpenAI; print('✅ OK')"

# Check trace IDs file
cat .trace_ids.json  # Should contain actual trace IDs, not empty strings

# Run single app manually to debug
source scripts/setup_environment.sh
.venv/bin/python3 tests/apps/unified_genai_test_app.py rag_pipeline
```

### Collector fails to start

**Cause:** Config file not found or already running.

**Solution:**
```bash
# Check if already running
ps aux | grep otelcol

# Stop existing collector
./scripts/stop_collector.sh

# Check logs
cat /tmp/otelcol.log

# Restart
./scripts/start_collector.sh
```

### Missing dependencies

**Solution:**
```bash
./scripts/setup_venv.sh
```

### Azure OpenAI not configured

**Solution:**
```bash
cp config/azure_openai.env.template config/azure_openai.env
# Edit with your credentials
source scripts/setup_environment.sh
```

---

## Reports

After running tests, reports are generated in `reports/`:

- **HTML:** `reports/html/comprehensive_report_*.html`
- **JSON:** `reports/json/comprehensive_report_*.json`
- **Logs:** `reports/test_execution.log`
- **Allure Results:** `reports/allure-results/` (raw data)

### Viewing Reports

**HTML Report:**
```bash
open reports/html/latest_report.html
```

**Allure Report (requires Allure CLI):**
```bash
# Install Allure (one-time)
brew install allure  # macOS
# or download from https://github.com/allure-framework/allure2/releases

# Generate and open Allure HTML report
allure serve reports/allure-results
```

The Allure report provides:
- Interactive test results with history
- Test execution timeline
- Detailed failure analysis
- Attachments (screenshots, logs)
- Trend charts

---

## Manual Execution

Run individual steps:

```bash
# Step 1: Environment
source scripts/setup_environment.sh

# Step 2: Collector
./scripts/start_collector.sh

# Step 3: Test apps (generates traces)
./scripts/run_apps.sh all 2

# Step 4: Tests (validates traces)
./scripts/run_tests.sh

# Step 5: Stop collector
./scripts/stop_collector.sh
```

---

## Key Files

- `requirements.txt` - Python dependencies
- `config/azure_openai.env` - Azure OpenAI credentials (gitignored)
- `.trace_ids.json` - Generated trace IDs from test apps
- `bin/otel-config-ai.yaml` - OpenTelemetry Collector config
- `tests/apps/unified_genai_test_app.py` - Test application
- `tests/test_tc_pi2_*.py` - Test suites

---

## Debug Commands

```bash
# Check virtual environment
.venv/bin/python3 --version
.venv/bin/pip list | grep -E "(langchain|opentelemetry)"

# Check environment variables
echo $AZURE_OPENAI_ENDPOINT
echo $SPLUNK_REALM

# Check collector
ps aux | grep otelcol
tail -f /tmp/otelcol.log

# Verify imports
.venv/bin/python3 -c "from langchain_openai import ChatOpenAI; from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter; print('✅ All OK')"
```

---

## Evaluation Metrics

### Overview

Evaluation metrics (bias, toxicity, hallucination) are automatically generated when the evaluation packages are installed. These metrics appear as additional spans and attributes in traces.

### Required Packages

The following packages are installed via `requirements.txt`:

```python
-e ../../util/opentelemetry-util-genai           # Core GenAI utilities
-e ../../util/opentelemetry-util-genai-evals     # Evaluation framework
-e ../../util/opentelemetry-util-genai-evals-deepeval  # Deepeval evaluators
```

### Configuration

Evaluations are configured in `scripts/setup_environment.sh`:

```bash
export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity),AgentInvocation(hallucination))"
export OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE="1.0"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event"
```

### Expected in Traces

**Evaluation Spans:**
- `gen_ai.evaluation.task` - Parent evaluation span
- `deepeval.bias.evaluation` - Bias metric
- `deepeval.toxicity.evaluation` - Toxicity metric
- `deepeval.hallucination.evaluation` - Hallucination metric

**Evaluation Attributes:**
- `gen_ai.evaluation.bias.score` (0.0-1.0)
- `gen_ai.evaluation.bias.passed` (boolean)
- `gen_ai.evaluation.toxicity.score` (0.0-1.0)
- `gen_ai.evaluation.toxicity.passed` (boolean)
- `gen_ai.evaluation.hallucination.score` (0.0-1.0)
- `gen_ai.evaluation.hallucination.passed` (boolean)

**AI Details Tab (Splunk APM):**
- Model name and version
- Token usage (input/output/total)
- Cost estimation
- Evaluation results summary
- Pass/fail status for each metric

### Troubleshooting

**Issue: No evaluation metrics in traces**

```bash
# Verify packages installed
python -c "import opentelemetry.util.genai.evals; print('✓ Evals installed')"
python -c "import opentelemetry.util.genai.evals.deepeval; print('✓ Deepeval installed')"

# Check environment variables
echo $OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS

# Reinstall if needed
pip install -r requirements.txt

# Re-run tests
./scripts/run_all.sh
```

---

## Architecture

```
Test Apps → OTel Collector → Splunk APM
    ↓
Trace IDs → .trace_ids.json
    ↓
Test Suites → Validate Traces → Reports
```

**Flow:**
1. Apps execute LLM scenarios and generate traces
2. Traces sent to OTel Collector → Splunk APM
3. Trace IDs captured to `.trace_ids.json`
4. Tests read trace IDs and validate in Splunk APM
5. Reports generated with results

---

For detailed information, see `config/README.md` for configuration details.
