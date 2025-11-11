# Alpha Release Testing

Manual testing framework for validating Alpha release AI observability features against customer documentation.

## 📁 Structure

```
alpha-release-testing/
├── config/
│   ├── .env                         # Main configuration
│   └── .env.{realm}.template        # Realm templates (lab0, rc0, us1)
├── tests/apps/                      # Test applications
│   ├── langchain_evaluation_app.py  # LangChain multi-agent (6 scenarios)
│   ├── langgraph_travel_planner_app.py  # LangGraph workflow (5 agents)
│   └── traceloop_travel_planner_app.py  # Traceloop translator
├── docs/
│   ├── ALPHA_RELEASE_TEST_PLAN.md   # Test plan with all use cases
│   └── TEST_EXECUTION_CHECKLIST.md  # Execution tracking
└── README.md                        # This file
```

## 🎯 Purpose

Validate customer documentation use cases:
- Instrument AI Applications (zero-code & code-based)
- LangChain/LangGraph instrumentation
- Traceloop SDK integration
- Configuration settings
- Splunk APM UI verification

## 🚀 Quick Start

### One-Time Setup

```bash
cd alpha-release-testing

# Run setup script (one time only)
./setup.sh

# Edit config/.env and verify your OPENAI_API_KEY
vim config/.env
```

### Run Tests (Automated)

```bash
# Run all tests once (includes both zero-code and manual modes)
./run_tests.sh

# Run only LangChain test
./run_tests.sh langchain

# Run LangGraph test (both zero-code and manual modes)
./run_tests.sh langgraph

# Run LangGraph with zero-code instrumentation only
./run_tests.sh langgraph_zerocode

# Run LangGraph with manual instrumentation only
./run_tests.sh langgraph_manual

# Run all tests continuously every 30 seconds
./run_tests.sh loop_30

# Run only LangChain test every 60 seconds
./run_tests.sh langchain loop_60

# Run only LangGraph test every 120 seconds
./run_tests.sh langgraph loop_120
```

The script automatically:
- Activates virtual environment
- Loads environment variables (with proper export)
- Runs selected test application(s)
- **LangGraph runs in BOTH modes**: Zero-code (opentelemetry-instrument) and Manual (hardcoded)
- Shows summary of results
- **Loop mode**: Runs continuously at specified intervals (Press Ctrl+C to stop)

---

## 📝 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install Dependencies

```bash
cd alpha-release-testing

# Create virtual environment
uv venv .venv-langchain
source .venv-langchain/bin/activate

# Install pip
uv pip install pip

# Install local Splunk packages
pip install -e ../../../../util/opentelemetry-util-genai --no-deps && \
pip install -e ../../../../util/opentelemetry-util-genai-emitters-splunk --no-deps && \
pip install -e ../../../../util/opentelemetry-util-genai-evals --no-deps && \
pip install -e ../../../../util/opentelemetry-util-genai-evals-deepeval && \
pip install -e ../../../../instrumentation-genai/opentelemetry-instrumentation-langchain/
```

### 2. Configure Environment

```bash
# Copy template and edit
cp config/.env.lab0.template config/.env
vim config/.env  # Add your OPENAI_API_KEY

# Export environment variables (important!)
set -a
source config/.env
set +a
```

### 3. Run Tests Manually

```bash
cd tests/apps

# LangChain evaluation (6 scenarios)
python langchain_evaluation_app.py

# LangGraph travel planner - Manual instrumentation (hardcoded)
python langgraph_travel_planner_app.py

# LangGraph travel planner - Zero-code instrumentation
opentelemetry-instrument python langgraph_travel_planner_app.py
```

## 📊 Verify in Splunk APM

1. Navigate to Splunk APM (lab0: https://app.lab0.signalfx.com)
2. Go to **APM → Agents**
3. Find your service: `alpha-release-test`
4. Verify:
   - Agent names appear correctly
   - Evaluation metrics visible
   - Token usage tracked
   - Trace hierarchy correct

## 📚 Documentation

- **Test Plan**: `docs/ALPHA_RELEASE_TEST_PLAN.md` - All test cases and use cases
- **Checklist**: `docs/TEST_EXECUTION_CHECKLIST.md` - Track execution progress
- **Test Apps**: `tests/apps/README.md` - Detailed app documentation

## 🔧 Troubleshooting

**Environment variables not loaded:**
```bash
# Verify environment is loaded
echo $OPENAI_API_KEY
echo $OTEL_SERVICE_NAME

# Reload if needed
source config/.env
```

**Import errors:**
```bash
# Verify virtual environment is active
which python  # Should show .venv-langchain/bin/python

# Reinstall packages if needed
pip install -e ../../../../instrumentation-genai/opentelemetry-instrumentation-langchain/
```

**No telemetry in Splunk:**
- Check OTEL Collector is running: `curl http://localhost:4317`
- Verify `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env`
- Check service name matches in Splunk APM

---

**Status**: Ready for manual testing  
**Environment**: lab0 (Splunk Observability Cloud)  
**Last Updated**: November 11, 2025
