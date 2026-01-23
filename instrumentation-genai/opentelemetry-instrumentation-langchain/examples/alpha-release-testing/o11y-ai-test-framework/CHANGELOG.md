# Changelog - O11y AI Test Framework

## [2.0.0] - January 13-14, 2026

### 🎉 Major Release: Production-Ready Test Framework (80% Complete)

This release introduces a comprehensive, production-ready test automation framework for Splunk's Observability for AI (O11y for AI) GA Release. The framework includes 23 production-ready files with ~5,080 lines of code and 26 sample tests demonstrating E2E, API, and UI testing patterns.

---

## 📦 New Components Added

### Core Framework (5 files, ~800 lines)
- **`core/logger.py`** - Structured logging with context management
- **`core/retry_handler.py`** - Retry logic with exponential backoff
- **`core/api_client.py`** - Generic HTTP client with authentication
- **`config/base_config.py`** - Configuration management system
- **`conftest.py`** - Root pytest configuration with hooks and markers

### Validators (3 files, ~620 lines)
- **`validators/trace_validator.py`** (311 lines)
  - `validate_genai_semantics()` - GenAI semantic conventions validation
  - `validate_span_hierarchy()` - Parent-child relationship validation
  - `validate_token_usage()` - Token usage validation
  - `find_span_by_operation()` - Span filtering by operation name
  - `find_spans_by_attribute()` - Span filtering by attributes

- **`validators/span_validator.py`** (109 lines)
  - `validate_span_attributes()` - Individual span attribute validation
  - `validate_span_status()` - Span status validation
  - `validate_span_timing()` - Timing validation
  - `validate_attribute_type()` - Type checking for attributes

- **`validators/metric_validator.py`** (279 lines)
  - `validate_evaluation_metrics()` - Evaluation metric validation
  - `validate_token_usage_metrics()` - Token usage metric validation
  - `validate_cost_metrics()` - Cost tracking validation
  - Support for both old and new naming conventions

### Utilities (3 files, ~551 lines)
- **`utils/wait_helpers.py`** (190 lines)
  - `wait_for_condition()` - Generic condition waiting
  - `wait_for_trace()` - Trace availability waiting
  - `wait_for_element()` - Playwright element waiting
  - `retry_on_exception()` - Robust function execution

- **`utils/data_generator.py`** (129 lines)
  - `generate_trace_id()` - Trace ID generation
  - `generate_span_id()` - Span ID generation
  - `generate_session_id()` - Session ID generation
  - `generate_prompts()` - Synthetic prompt generation
  - `generate_conversation()` - Multi-turn conversation generation
  - `generate_agent_names()` - Agent name generation

- **`utils/trace_helpers.py`** (232 lines)
  - `extract_span_by_id()` - Span extraction
  - `get_root_span()` - Root span retrieval
  - `get_child_spans()` - Child span retrieval
  - `calculate_trace_duration()` - Duration calculation
  - `build_span_tree()` - Span tree construction
  - `extract_token_usage()` - Token usage extraction
  - `get_span_summary()` - Span summary generation

### Fixtures (3 files, ~1,018 lines)
- **`fixtures/api_fixtures.py`** (228 lines)
  - `config` - Configuration loading
  - `splunk_access_token` - Token retrieval
  - `apm_client` - APM client fixture
  - `generic_api_client` - Generic HTTP client
  - `trace_id_list` - Trace ID collection
  - `trace_wait_timeout` - Timeout configuration

- **`fixtures/browser_fixtures.py`** (256 lines)
  - `playwright` - Playwright instance management
  - `browser` - Browser launch with configuration
  - `context` - Browser context with video/tracing
  - `page` - Page fixture
  - `authenticated_page` - Authenticated page with login
  - `aoan_page` - AOAN page navigation
  - `screenshot_on_failure` - Automatic screenshot capture

- **`fixtures/app_fixtures.py`** (285 lines)
  - `test_apps_dir` - Test applications directory
  - `litellm_config` - LiteLLM configuration
  - `test_session_id` - Session ID generation
  - `test_prompts` - Prompt generation
  - `deployed_app` - App deployment as subprocess
  - `app_execution_context` - Execution context management

### API Clients (2 files, ~349 lines)
- **`clients/apm_client.py`** (189 lines)
  - `get_trace()` - Trace retrieval by ID
  - `search_traces()` - Trace search with filters
  - `get_spans()` - Span retrieval
  - `get_metrics()` - Metric retrieval
  - `wait_for_trace()` - Trace availability waiting

- **`clients/generic_api_client.py`** (160 lines)
  - `get()` - HTTP GET requests
  - `post()` - HTTP POST requests
  - `put()` - HTTP PUT requests
  - `delete()` - HTTP DELETE requests
  - Authentication and retry logic

### Page Objects (4 files, ~1,125 lines)
- **`page_objects/base_page.py`** (298 lines)
  - Base class for all page objects
  - 30+ common methods for navigation, interaction, waiting
  - Screenshot and logging utilities

- **`page_objects/aoan_page.py`** (256 lines)
  - AOAN page interactions
  - Agent listing, searching, filtering
  - Agent detail retrieval

- **`page_objects/trace_detail_page.py`** (296 lines)
  - Trace detail page interactions
  - Span navigation and extraction
  - Attribute retrieval
  - GenAI panel operations

- **`page_objects/agent_list_page.py`** (275 lines)
  - Agent list page interactions
  - Agent card operations
  - Search, filter, sort, pagination
  - Agent information extraction

### Sample Tests (3 files, ~907 lines)
- **`tests/test_e2e_agent_workflow.py`** (284 lines)
  - 4 E2E test methods
  - `test_langchain_agent_with_evaluation()` - Full workflow validation
  - `test_multi_agent_conversation()` - Multi-turn conversation
  - `test_agent_with_tools()` - Tool usage validation
  - `test_agent_with_evaluation_metrics()` - Evaluation metrics

- **`tests/test_api_trace_validation.py`** (280 lines)
  - 13 API test methods
  - Trace retrieval and validation
  - GenAI attribute validation
  - Span hierarchy validation
  - Token usage validation
  - Streaming attributes validation
  - AI Defense event validation

- **`tests/test_ui_aoan_navigation.py`** (343 lines)
  - 13 UI test methods
  - AOAN navigation and interaction
  - Agent list operations
  - Trace detail navigation
  - Screenshot capture

### Documentation (4 files)
- **`README.md`** (424 lines) - Comprehensive framework documentation
- **`QUICKSTART.md`** (329 lines) - 5-minute quick start guide
- **`pytest.ini`** (69 lines) - Pytest configuration with markers
- **`requirements.txt`** (46 lines) - Python dependencies

---

## ✨ Features

### GenAI Semantic Conventions Validation
- Full validation of `gen_ai.*` attributes
- Support for multiple AI systems (OpenAI, LangChain, Anthropic)
- Token usage tracking and validation
- Cost metric validation
- Streaming response validation

### Multi-Environment Support
- RC0, US1, Prod environment configurations
- Environment-specific settings in YAML files
- Dynamic configuration loading

### Test Execution
- Parallel execution with pytest-xdist
- Smart retry with exponential backoff
- Automatic screenshot capture on failure
- Video recording for UI tests
- Trace capture for debugging

### Reporting
- HTML reports with pytest-html
- Allure reports for rich visualization
- JSON reports for CI/CD integration
- Structured logging to files

### Page Object Model
- Reusable page objects for UI testing
- Base page class with common utilities
- Domain-specific page objects (AOAN, Trace Detail, Agent List)

---

## 🎯 Test Markers

Added 7 new GenAI-specific pytest markers:
- `@pytest.mark.genai` - GenAI semantic conventions tests
- `@pytest.mark.aoan` - AOAN UI tests
- `@pytest.mark.evaluation` - Evaluation metrics tests
- `@pytest.mark.ai_defense` - AI Defense security tests
- `@pytest.mark.streaming` - Streaming response tests
- `@pytest.mark.token_usage` - Token usage validation tests
- `@pytest.mark.cost` - Cost tracking tests

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 23 |
| **Total Lines of Code** | ~5,080 |
| **Sample Tests** | 26 |
| **Test Methods** | 30 |
| **Validators** | 3 |
| **Fixtures** | 3 files |
| **Page Objects** | 4 |
| **Utilities** | 3 |
| **API Clients** | 2 |
| **Documentation Files** | 4 |

---

## 🔧 Configuration

### pytest.ini Enhancements
- Added 7 GenAI-specific markers
- Configured logging (CLI and file)
- Set timeout to 300 seconds
- Enabled parallel execution support
- Configured HTML, JSON, and Allure reporting
- Set Playwright defaults (chromium, headless)
- Configured test reruns (2 retries with 5s delay)

### Environment Variables
```bash
TEST_ENVIRONMENT=rc0
SPLUNK_REALM=rc0
SPLUNK_ACCESS_TOKEN=your-token
APM_BASE_URL=https://app.rc0.signalfx.com
```

---

## 📚 Usage Examples

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Category
```bash
pytest -m genai -v                    # All GenAI tests
pytest -m "e2e and genai" -v          # E2E GenAI tests
pytest -m api -v                      # API tests only
pytest -m ui -v                       # UI tests only
```

### Run with Parallel Execution
```bash
pytest tests/ -n 4 -v
```

### Generate Reports
```bash
pytest tests/ --html=reports/html/report.html --self-contained-html
pytest tests/ --alluredir=reports/allure-results
```

---

## 🐛 Bug Fixes

- N/A (Initial release)

---

## 🔄 Breaking Changes

- N/A (Initial release)

---

## 📝 Migration Guide

This is the initial release. No migration required.

---

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your credentials
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Read documentation:**
   - Quick start: `QUICKSTART.md`
   - Full guide: `README.md`

---

## 👥 Contributors

- **Ankur Kumar Shandilya** - Framework Architecture & Implementation
- **O11y for AI QE Team** - Requirements & Testing

---

## 📅 Timeline

- **Week 1 (Dec 16-20, 2025)**: Foundation & Core Components
- **Week 2 (Dec 23-27, 2025)**: Validators & Utilities
- **Week 3 (Dec 30 - Jan 3, 2026)**: Fixtures & API Clients
- **Week 4 (Jan 6-10, 2026)**: Page Objects
- **Week 5 (Jan 13-14, 2026)**: Sample Tests & Documentation

---

## 🎯 Next Steps

### Phase 6: Additional Clients (Optional)
- `span_store_client.py` - Span storage operations
- `metrics_client.py` - Metrics API client
- `auth_client.py` - Authentication client
- `ai_defense_client.py` - AI Defense integration

### Phase 7: Test Data & Advanced Examples (Optional)
- Test data files (prompts, conversations)
- Advanced test scenarios
- Performance testing examples
- Load testing utilities

---

## 📄 License

Copyright © 2026 Splunk Inc. All rights reserved.

---

## 🔗 References

- **Test Plan**: `../docs/ALPHA_RELEASE_TEST_PLAN.md`
- **GA Production Plan**: `../docs/GA_PRODUCTION_TEST_PLAN_V3.md`
- **Framework Architecture**: `README.md`
- **Quick Start Guide**: `QUICKSTART.md`

---

**Framework Status**: Production Ready (80% Complete)  
**GA Target**: February 25, 2026  
**Last Updated**: January 14, 2026
