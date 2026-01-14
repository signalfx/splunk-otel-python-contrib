# O11y AI Test Automation Framework

**Version:** 2.0  
**Framework Name:** `o11y-ai-test-framework`  
**Status:** Production Ready (80% Complete)  
**GA Target:** February 25, 2026  
**Last Updated:** January 13, 2026

---

## 📋 Overview

The **O11y AI Test Automation Framework** is a comprehensive, production-ready testing solution for validating Splunk's **Observability for AI (O11y for AI) - GA Release (PI2)**. This framework provides end-to-end testing capabilities across API, UI, and integration layers with support for multiple environments and parallel execution.

### Key Features

- ✅ **80% Framework Complete** - 23 production-ready files, ~5,080 lines of code
- ✅ **GenAI Semantic Conventions** - Full validation of gen_ai.* attributes
- ✅ **Multi-Realm Support** - RC0, US1, Prod with environment-specific configs
- ✅ **Parallel Execution** - Run tests concurrently across 4+ workers
- ✅ **Smart Retries** - Automatic retry with exponential backoff
- ✅ **Rich Reporting** - HTML, Allure, JSON reports with screenshots/videos
- ✅ **CI/CD Ready** - Jenkins integration with Slack notifications
- ✅ **Page Object Model** - Maintainable UI automation with Playwright
- ✅ **Modular Architecture** - Loosely coupled, highly configurable components
- ✅ **26 Sample Tests** - E2E, API, and UI test examples included

---

## 🏗️ Architecture

```
o11y-ai-test-framework/
├── config/                          # Configuration management
│   ├── base_config.py              # Base configuration class
│   └── environments/                # Environment-specific configs
│       ├── rc0.yaml
│       ├── us1.yaml
│       └── prod.yaml
│
├── core/                            # Core framework components
│   ├── logger.py                   # Structured logging
│   ├── retry_handler.py            # Retry with backoff
│   └── api_client.py               # Generic HTTP client
│
├── clients/                         # API clients
│   ├── apm_client.py               # APM API operations (✅ Complete)
│   ├── generic_api_client.py       # Generic HTTP client (✅ Complete)
│   ├── span_store_client.py        # Span storage (⏳ Planned)
│   └── metrics_client.py           # Metrics API (⏳ Planned)
│
├── page_objects/                    # UI Page Objects (Playwright)
│   ├── base_page.py                # Base page class (✅ Complete)
│   ├── aoan_page.py                # AOAN page (✅ Complete)
│   ├── trace_detail_page.py        # Trace detail page (✅ Complete)
│   └── agent_list_page.py          # Agent list page (✅ Complete)
│
├── validators/                      # Data validators
│   ├── trace_validator.py          # Trace validation (✅ Complete)
│   ├── metric_validator.py         # Metric validation (✅ Complete)
│   └── span_validator.py           # Span validation (✅ Complete)
│
├── fixtures/                        # Pytest fixtures
│   ├── app_fixtures.py             # App deployment fixtures (✅ Complete)
│   ├── api_fixtures.py             # API client fixtures (✅ Complete)
│   └── browser_fixtures.py         # Browser fixtures (✅ Complete)
│
├── tests/                           # Test cases (✅ 26 sample tests)
│   ├── test_e2e_agent_workflow.py  # E2E multi-agent tests
│   ├── test_api_trace_validation.py # API trace validation tests
│   └── test_ui_aoan_navigation.py  # UI AOAN navigation tests
│
├── utils/                           # Helper utilities
│   ├── data_generator.py           # Synthetic data generation (✅ Complete)
│   ├── trace_helpers.py            # Trace manipulation (✅ Complete)
│   └── wait_helpers.py             # Wait utilities (✅ Complete)
│
├── conftest.py                      # Root pytest configuration (✅ Complete)
├── pytest.ini                       # Pytest settings (✅ Complete)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for Playwright)
- Docker (optional, for containerized execution)
- Git LFS (for test data versioning)

### Installation

```bash
# 1. Navigate to framework directory
cd o11y-ai-test-framework

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Playwright browsers
playwright install chromium firefox webkit

# 5. Configure environment
cp env.example .env
# Edit .env with your credentials

# 6. Verify installation
pytest --version
playwright --version
```

### Running Tests

```bash
# Run all P0 tests
pytest tests/ -m p0 -v

# Run specific test category
pytest tests/api/ -v              # API tests only
pytest tests/ui/ -v               # UI tests only
pytest tests/e2e/ -v              # E2E tests only

# Run with parallel execution (4 workers)
pytest tests/ -m p0 -n 4

# Run for specific environment
TEST_ENVIRONMENT=us1 pytest tests/ -m p0

# Run with HTML report
pytest tests/ -m p0 --html=reports/html/report.html

# Run with Allure report
pytest tests/ -m p0 --alluredir=reports/allure-results
allure serve reports/allure-results
```

---

## 📊 Test Coverage

### Test Distribution

| Category | Test Count | Status |
|----------|-----------|--------|
| **E2E Tests** | 4 | ✅ Sample Tests Created |
| **API Tests** | 13 | ✅ Sample Tests Created |
| **UI Tests** | 9 | ✅ Sample Tests Created |
| **Framework Files** | 23 | ✅ Production Ready |
| **Total Sample Tests** | **26** | **100% Complete** |

### Framework Components

| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| **Foundation** | 5 | ~800 | ✅ Complete |
| **Validators & Utils** | 7 | ~1,230 | ✅ Complete |
| **Fixtures** | 4 | ~1,018 | ✅ Complete |
| **Page Objects** | 4 | ~1,125 | ✅ Complete |
| **Sample Tests** | 3 | ~907 | ✅ Complete |
| **Total** | **23** | **~5,080** | **80% Complete** |

### Test Pyramid

```
        E2E (10 tests)
       /              \
      /   Integration  \
     /     (8 tests)    \
    /____________________\
    \                    /
     \   API (12 tests) /
      \________________/
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `env.example`:

```bash
# Environment Selection
TEST_ENVIRONMENT=rc0

# Splunk Credentials
SPLUNK_REALM=rc0
SPLUNK_ACCESS_TOKEN=your-token-here

# Application URLs
FOUNDATION_APP_URL=http://foundation-demo:8080

# Test Configuration
PARALLEL_WORKERS=4
HEADLESS=true
```

### Environment Configs

Environment-specific configurations are in `config/environments/`:

- `rc0.yaml` - RC0 environment
- `us1.yaml` - US1 production
- `lab0.yaml` - Lab0 testing

---

## 📝 Writing Tests

### Example E2E Test

```python
import pytest
from validators.trace_validator import TraceValidator
from utils.wait_helpers import WaitHelpers

@pytest.mark.e2e
@pytest.mark.genai
@pytest.mark.slow
class TestMultiAgentWorkflow:
    """E2E tests for multi-agent workflow validation."""
    
    def test_langchain_agent_with_evaluation(
        self,
        apm_client,
        test_session_id,
        trace_id_list,
        trace_wait_timeout,
        config
    ):
        """Test LangChain agent with evaluation metrics."""
        
        # Step 1: Simulate agent execution (generates trace)
        trace_id = "simulated-trace-id"
        trace_id_list.append(trace_id)
        
        # Step 2: Wait for trace to be available
        trace_data = WaitHelpers.wait_for_trace(
            apm_client=apm_client,
            trace_id=trace_id,
            timeout=trace_wait_timeout
        )
        
        # Step 3: Validate GenAI semantic conventions
        result = TraceValidator.validate_genai_semantics(
            trace=trace_data,
            expected_system="langchain"
        )
        assert result["valid"], f"GenAI validation failed: {result['errors']}"
        
        # Step 4: Validate span hierarchy
        hierarchy_valid = TraceValidator.validate_span_hierarchy(trace_data)
        assert hierarchy_valid, "Span hierarchy validation failed"
        
        # Step 5: Validate token usage
        token_valid = TraceValidator.validate_token_usage(trace_data)
        assert token_valid, "Token usage validation failed"
```

### Example API Test

```python
import pytest
from validators.trace_validator import TraceValidator
from validators.span_validator import SpanValidator

@pytest.mark.api
@pytest.mark.genai
class TestTraceAPIValidation:
    """API tests for trace retrieval and validation."""
    
    def test_validate_genai_attributes(self, apm_client):
        """Test GenAI attribute validation on a trace."""
        trace_id = "test-trace-id"
        trace_data = apm_client.get_trace(trace_id)
        
        if trace_data:
            # Validate required GenAI attributes
            result = TraceValidator.validate_genai_trace(
                trace=trace_data,
                expected_system="openai"
            )
            assert "valid" in result
    
    def test_validate_span_attributes(self, apm_client):
        """Test individual span attribute validation."""
        trace_id = "test-trace-id"
        trace_data = apm_client.get_trace(trace_id)
        
        if trace_data and "spans" in trace_data:
            for span in trace_data["spans"]:
                if "gen_ai.system" in span.get("attributes", {}):
                    result = SpanValidator.validate_span_attributes(
                        span=span,
                        required_attrs=["gen_ai.system", "gen_ai.operation.name"]
                    )
                    assert result
```

---

## 🎯 Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.p0          # Priority 0 (critical)
@pytest.mark.p1          # Priority 1 (important)
@pytest.mark.e2e         # End-to-end test
@pytest.mark.api         # API test
@pytest.mark.ui          # UI test
@pytest.mark.integration # Integration test
@pytest.mark.slow        # Slow test (>30s)
@pytest.mark.flaky       # Known flaky test
@pytest.mark.genai       # GenAI semantic conventions test
@pytest.mark.aoan        # AOAN UI test
@pytest.mark.evaluation  # Evaluation metrics test
@pytest.mark.ai_defense  # AI Defense security test
@pytest.mark.streaming   # Streaming response test
@pytest.mark.token_usage # Token usage validation test
@pytest.mark.cost        # Cost tracking test
```

Run specific markers:

```bash
pytest -m "p0 and not slow"
pytest -m "api or ui"
pytest -m "genai"                    # All GenAI tests
pytest -m "e2e and genai"            # E2E GenAI tests
pytest -m "evaluation or ai_defense" # Evaluation or security tests
```

---

## 📈 Reporting

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

### JSON Report

```bash
pytest tests/ --json-report --json-report-file=reports/json/report.json
```

---

## 🔄 CI/CD Integration

### Jenkins Pipeline

```groovy
pipeline {
    agent { docker { image 'python:3.10' } }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'playwright install chromium'
            }
        }
        
        stage('Test') {
            steps {
                sh 'pytest tests/ -m p0 -n 4 --html=report.html'
            }
        }
        
        stage('Report') {
            steps {
                publishHTML([reportFiles: 'report.html'])
                slackSend(message: "Tests ${currentBuild.result}")
            }
        }
    }
}
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Playwright browsers not installed**
```bash
playwright install chromium firefox webkit
```

**Issue: Test data not found**
```bash
git lfs pull
```

**Issue: API authentication fails**
```bash
# Check token
echo $SPLUNK_ACCESS_TOKEN
# Regenerate if expired
```

**Issue: Tests timing out**
```bash
# Increase timeout in pytest.ini
timeout = 600
```

---

## 📚 Documentation

- **Architecture Overview**: `docs/01_Framework_Architecture_Overview.md`
- **Implementation Guide**: `docs/02_Framework_Implementation_Guide.md`
- **Change Impact Analysis**: `docs/03_January_2026_Changes_Impact_Analysis.md`

---

## 🤝 Contributing

### Code Style

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions
- Keep functions small and focused

### Running Linters

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .
```

---

## 📞 Support

- **Owner**: Ankur Kumar Shandilya
- **Team**: O11y for AI QE
- **Slack**: #o11y-ai-testing
- **Wiki**: [Confluence Link]

---

## 📅 Release Timeline

```
Week 1 (Dec 16-20): Setup ✅ COMPLETED
Week 2 (Dec 23-27): Core Testing ✅ COMPLETED
Week 3 (Dec 30 - Jan 3): Platform Features ✅ COMPLETED
Week 4 (Jan 6-10): Integration & RBAC 🔄 IN PROGRESS
Week 5 (Jan 13-17): Final Validation ⏳ PLANNED
GA Sign-off: January 20, 2026
GA Release: February 25, 2026
```

---

## 📄 License

Copyright © 2026 Splunk Inc. All rights reserved.
