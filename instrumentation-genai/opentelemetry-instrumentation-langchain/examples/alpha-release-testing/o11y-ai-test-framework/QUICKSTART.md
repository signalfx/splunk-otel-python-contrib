# 🚀 Quick Start Guide - O11y AI Test Framework

**Get up and running in 5 minutes!**

---

## ⚡ Installation (5 minutes)

### Step 1: Clone & Navigate
```bash
cd o11y-ai-test-framework
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

### Step 4: Configure Environment
```bash
# Copy example environment file
cp env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

**Required environment variables:**
```bash
TEST_ENVIRONMENT=rc0
SPLUNK_REALM=rc0
SPLUNK_ACCESS_TOKEN=your-token-here
APM_BASE_URL=https://app.rc0.signalfx.com
```

---

## 🧪 Run Your First Test (1 minute)

### Run All Sample Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# E2E tests only
pytest tests/test_e2e_agent_workflow.py -v

# API tests only
pytest tests/test_api_trace_validation.py -v

# UI tests only
pytest tests/test_ui_aoan_navigation.py -v
```

### Run with Markers
```bash
# All GenAI tests
pytest -m genai -v

# E2E GenAI tests
pytest -m "e2e and genai" -v

# Slow tests excluded
pytest -m "not slow" -v
```

---

## 📊 Generate Reports

### HTML Report
```bash
pytest tests/ --html=reports/html/report.html --self-contained-html
open reports/html/report.html
```

### Allure Report
```bash
pytest tests/ --alluredir=reports/allure-results
allure serve reports/allure-results
```

---

## 🎯 Sample Test Overview

### 1. E2E Multi-Agent Workflow Test
**File:** `tests/test_e2e_agent_workflow.py`  
**Tests:** 4 test methods  
**What it does:**
- Validates LangChain agent execution with evaluation metrics
- Tests multi-agent conversation flows
- Validates tool usage spans
- Checks evaluation metric formats

**Run it:**
```bash
pytest tests/test_e2e_agent_workflow.py::TestMultiAgentWorkflow::test_langchain_agent_with_evaluation -v
```

### 2. API Trace Validation Test
**File:** `tests/test_api_trace_validation.py`  
**Tests:** 13 test methods  
**What it does:**
- Retrieves traces via APM API
- Validates GenAI semantic conventions (gen_ai.*)
- Checks span attributes and hierarchy
- Validates token usage and costs
- Tests streaming attributes
- Validates AI Defense events

**Run it:**
```bash
pytest tests/test_api_trace_validation.py::TestTraceAPIValidation::test_validate_genai_attributes -v
```

### 3. UI AOAN Navigation Test
**File:** `tests/test_ui_aoan_navigation.py`  
**Tests:** 13 test methods  
**What it does:**
- Navigates to AOAN page
- Lists and searches agents
- Retrieves agent details
- Navigates to trace details
- Tests UI interactions with Playwright

**Run it:**
```bash
pytest tests/test_ui_aoan_navigation.py::TestAOANNavigation::test_navigate_to_aoan_page -v
```

---

## 🔧 Common Commands

### Parallel Execution (4 workers)
```bash
pytest tests/ -n 4 -v
```

### Run with Specific Environment
```bash
TEST_ENVIRONMENT=us1 pytest tests/ -v
```

### Run with Video Recording
```bash
pytest tests/test_ui_aoan_navigation.py --video=on
```

### Run with Browser Visible (non-headless)
```bash
pytest tests/test_ui_aoan_navigation.py --headed
```

### Run with Slow Motion (for debugging)
```bash
pytest tests/test_ui_aoan_navigation.py --slowmo=1000
```

---

## 📁 Framework Structure

```
o11y-ai-test-framework/
├── tests/                           # 26 sample tests
│   ├── test_e2e_agent_workflow.py  # 4 E2E tests
│   ├── test_api_trace_validation.py # 13 API tests
│   └── test_ui_aoan_navigation.py  # 9 UI tests
│
├── validators/                      # Validation utilities
│   ├── trace_validator.py          # Trace & GenAI validation
│   ├── span_validator.py           # Span validation
│   └── metric_validator.py         # Metric validation
│
├── fixtures/                        # Pytest fixtures
│   ├── api_fixtures.py             # API clients & config
│   ├── browser_fixtures.py         # Playwright setup
│   └── app_fixtures.py             # App deployment
│
├── page_objects/                    # UI page objects
│   ├── base_page.py                # Base page class
│   ├── aoan_page.py                # AOAN page
│   ├── trace_detail_page.py        # Trace detail page
│   └── agent_list_page.py          # Agent list page
│
├── utils/                           # Helper utilities
│   ├── data_generator.py           # Synthetic data
│   ├── trace_helpers.py            # Trace manipulation
│   └── wait_helpers.py             # Wait utilities
│
├── clients/                         # API clients
│   ├── apm_client.py               # APM API
│   └── generic_api_client.py       # Generic HTTP
│
├── conftest.py                      # Root pytest config
├── pytest.ini                       # Pytest settings
├── requirements.txt                 # Dependencies
└── README.md                        # Full documentation
```

---

## 🎓 Writing Your First Test

### Example: Simple API Test
```python
import pytest
from validators.trace_validator import TraceValidator

@pytest.mark.api
@pytest.mark.genai
def test_my_trace_validation(apm_client):
    """My first trace validation test."""
    
    # Retrieve trace
    trace_id = "your-trace-id"
    trace_data = apm_client.get_trace(trace_id)
    
    # Validate GenAI attributes
    result = TraceValidator.validate_genai_trace(
        trace=trace_data,
        expected_system="openai"
    )
    
    assert result["valid"], f"Validation failed: {result['errors']}"
```

**Save as:** `tests/test_my_first_test.py`

**Run it:**
```bash
pytest tests/test_my_first_test.py -v
```

---

## 🐛 Troubleshooting

### Issue: Playwright browsers not installed
```bash
playwright install chromium firefox webkit
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Tests timing out
```bash
# Increase timeout in pytest.ini
timeout = 600
```

### Issue: Authentication fails
```bash
# Check your token
echo $SPLUNK_ACCESS_TOKEN

# Verify realm
echo $SPLUNK_REALM
```

### Issue: No traces found
```bash
# Check APM base URL
echo $APM_BASE_URL

# Verify trace ID format
# Should be 32-character hex string
```

---

## 📚 Next Steps

1. **Read Full Documentation**: `README.md`
2. **Explore Sample Tests**: `tests/` directory
3. **Understand Validators**: `validators/` directory
4. **Learn Page Objects**: `page_objects/` directory
5. **Review Fixtures**: `fixtures/` directory

---

## 🎯 Test Markers Reference

| Marker | Description | Example |
|--------|-------------|---------|
| `@pytest.mark.e2e` | End-to-end test | `pytest -m e2e` |
| `@pytest.mark.api` | API test | `pytest -m api` |
| `@pytest.mark.ui` | UI test | `pytest -m ui` |
| `@pytest.mark.genai` | GenAI validation | `pytest -m genai` |
| `@pytest.mark.aoan` | AOAN UI test | `pytest -m aoan` |
| `@pytest.mark.evaluation` | Evaluation metrics | `pytest -m evaluation` |
| `@pytest.mark.ai_defense` | AI Defense security | `pytest -m ai_defense` |
| `@pytest.mark.slow` | Slow test (>30s) | `pytest -m "not slow"` |

---

## 💡 Pro Tips

1. **Use `-v` flag** for verbose output
2. **Use `-s` flag** to see print statements
3. **Use `--pdb`** to drop into debugger on failure
4. **Use `-k pattern`** to run tests matching pattern
5. **Use `--lf`** to run last failed tests only
6. **Use `--sw`** to stop on first failure

---

## 📞 Support

- **Documentation**: `README.md`
- **Sample Tests**: `tests/` directory
- **Issues**: Check logs in `reports/test_execution.log`

---

## ✅ Verification Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list | grep pytest`)
- [ ] Playwright browsers installed (`playwright --version`)
- [ ] Environment variables configured (`.env` file)
- [ ] Sample tests run successfully (`pytest tests/ -v`)
- [ ] Reports generated (`reports/html/report.html`)

---

**🎉 You're ready to start testing!**

Run `pytest tests/ -v` to see all 26 sample tests in action.
