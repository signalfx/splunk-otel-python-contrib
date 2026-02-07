# Tokenator Implementation Plan

## Project Overview

**Package Name**: `opentelemetry-util-genai-rate-limit-predictor`  
**Codename**: Tokenator  
**Goal**: Predict rate limit breaches at span, trace, and workflow levels with SQLite persistence

**Tagline**: *"I'll be back... before you hit your rate limit"*

---

## Phase 1: Foundation (Day 1)

### 1.1 Create Package Structure

```
util/opentelemetry-util-genai-rate-limit-predictor/
├── pyproject.toml
├── README.rst
├── CHANGELOG.md
├── LICENSE
├── pytest.ini
├── src/
│   └── opentelemetry/
│       └── util/
│           └── genai/
│               └── rate_limit/
│                   ├── __init__.py
│                   ├── version.py
│                   ├── emitter.py
│                   ├── tracker.py
│                   ├── predictor.py
│                   ├── models.py
│                   └── providers/
│                       ├── __init__.py
│                       ├── base.py
│                       └── openai.py
└── tests/
    ├── __init__.py
    ├── test_tracker.py
    ├── test_predictor.py
    ├── test_emitter.py
    └── test_providers.py
```

### 1.2 Core Files to Create

**Priority Order:**
1. `pyproject.toml` - Package configuration
2. `version.py` - Version management
3. `models.py` - SQLite schema
4. `tracker.py` - Token tracking with SQLite
5. `providers/base.py` + `providers/openai.py` - Rate limit API clients
6. `predictor.py` - Prediction algorithms
7. `emitter.py` - Main emitter implementation
8. `__init__.py` - Package exports

---

## Phase 2: Core Implementation (Day 1-2)

### 2.1 SQLite Database Layer (`models.py` + `tracker.py`)

**Tasks:**
- [ ] Define database schema (token_usage, trace_token_usage, workflow_patterns)
- [ ] Implement `SQLiteTokenTracker` class
- [ ] Add trace-level aggregation methods
- [ ] Add workflow pattern learning (exponential moving average)
- [ ] Implement cleanup/retention logic
- [ ] Add thread-safe connection management

**Window Semantics:**
All time windows are **rolling** (not calendar-aligned):
- **TPM**: sum of tokens in the last 60 seconds from `now`.
- **Weekly**: sum of tokens in the last 7 × 86 400 seconds from `now`.
- **Monthly**: sum of tokens in the last 30 × 86 400 seconds from `now`.

This simplifies queries (`WHERE timestamp >= :now - :window_seconds`) and avoids timezone/locale issues. Calendar-aligned windows (e.g., "this calendar month") can be added as a future enhancement.

**Key Methods:**
```python
- record() - Record token usage with trace correlation
- get_trace_usage() - Get aggregated trace tokens
- get_current_tpm() - Sum tokens WHERE timestamp >= now - 60
- get_weekly_usage() - Sum tokens WHERE timestamp >= now - 604800
- get_monthly_usage() - Sum tokens WHERE timestamp >= now - 2592000
- mark_trace_complete() - Finalize trace and update patterns
- get_workflow_pattern() - Get learned pattern
```

### 2.2 Provider API Clients (`providers/`)

**Tasks:**
- [ ] Create `RateLimitProvider` abstract base class
- [ ] Implement `OpenAIRateLimitProvider` with hardcoded free tier limits
- [ ] Add async support for future API integration
- [ ] Support multiple models (gpt-4o-mini, gpt-4.1, etc.)

**Initial Limits (Hackathon - Hardcoded Free Tier, as of 2026-02-06):**
```python
GPT-4o mini:
  - RPM: 30
  - TPM: 200,000
  - Monthly input: 500M
  - Monthly completion: 50M
  - Weekly: 100M

GPT-4.1:
  - RPM: 15
  - TPM: 1,000,000
  - Monthly input: 50M
  - Monthly completion: 5M
  - Weekly: 15M
```

**Note**: For hackathon purposes, limits are hardcoded based on OpenAI free tier documentation. In production, these should be:
- Fetched from provider management APIs
- Configurable via environment variables
- Overridable per model/account tier

**Future Enhancement**: Add support for:
- Provider API integration (OpenAI, Anthropic, etc.)
- Environment variable overrides: `OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_OPENAI_TPM=200000`
- Configuration file support for custom limits

### 2.3 Prediction Engine (`predictor.py`)

**Tasks:**
- [ ] Implement `MultiLimitPredictor` class
- [ ] Add TPM prediction
- [ ] Add weekly limit prediction
- [ ] Add monthly limit prediction
- [ ] Add RPM prediction
- [ ] Implement `WorkflowCompletionPredictor`
- [ ] Add progress calculation (current / predicted total)
- [ ] Generate actionable recommendations

**Key Methods:**
```python
- predict_all_limits() - Predict all limit types
- predict_workflow_completion() - Predict if workflow can complete
- _predict_tpm() - TPM-specific prediction
- _predict_weekly() - Weekly limit prediction
- _predict_monthly() - Monthly limit prediction
- _generate_recommendation() - Create actionable advice
```

### 2.4 Emitter Implementation (`emitter.py`)

**Tasks:**
- [ ] Implement `RateLimitPredictorEmitter` class
- [ ] Handle `on_start()` for workflow tracking
- [ ] Handle `on_end()` for LLM invocations
- [ ] Extract trace_id and span_id from GenAI objects
- [ ] Call tracker to record usage
- [ ] Call predictor to generate predictions
- [ ] Emit OpenTelemetry events for warnings
- [ ] Implement `load_emitters()` entry point

**Trace/Span ID Source:**
The `GenAI` base dataclass (in `types.py`) exposes `trace_id: Optional[int]` and `span_id: Optional[int]` fields, populated by `store_span_context()` from `span_context.py`. The emitter reads these directly:
```python
trace_id: str = f"{obj.trace_id:032x}" if obj.trace_id else None
span_id:  str = f"{obj.span_id:016x}"  if obj.span_id  else None
```
Fallback: if `trace_id` is not set, call `extract_span_context(obj.span)` → `store_span_context(obj, ctx)` to populate them from the live OTel span (mirrors the pattern in `handler.py:start_llm`).

**Key Methods:**
```python
- on_start() - Track workflow start
- on_end() - Process LLM/workflow end
- _handle_llm_end() - Process LLM invocation
- _handle_workflow_end() - Finalize workflow
- _get_trace_id() - Read obj.trace_id (int) → format as 32-char hex
- _get_span_id() - Read obj.span_id (int) → format as 16-char hex
- _emit_workflow_warning() - Emit warning event
```

---

## Phase 3: Integration & Testing (Day 2-3)

### 3.1 Unit Tests

**Test Files:**
- `test_tracker.py` - Test SQLite operations, aggregation, patterns
- `test_predictor.py` - Test prediction algorithms
- `test_emitter.py` - Test emitter lifecycle
- `test_providers.py` - Test rate limit fetching

**Key Test Cases:**
```python
# Tracker tests
- test_record_token_usage()
- test_trace_aggregation()
- test_workflow_pattern_learning()
- test_multi_window_queries()
- test_cleanup_old_data()

# Predictor tests
- test_tpm_prediction()
- test_weekly_prediction()
- test_monthly_prediction()
- test_workflow_completion_prediction()
- test_recommendation_generation()

# Emitter tests
- test_emitter_registration()
- test_emitter_comma_separated_config()  # Validates loading via "span_metric_event,rate_limit_predictor"
- test_on_end_processing()
- test_trace_id_extraction()
- test_event_emission()
```

### 3.2 Integration with Existing System

**Tasks:**
- [ ] Verify entry point registration in `pyproject.toml`
- [ ] Test with `opentelemetry-util-genai` core
- [ ] Validate emitter discovery via entry points
- [ ] Test with multi-agent travel planner example

**Configuration:**
```bash
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"
export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_WARNING_THRESHOLD=0.8
export OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_DB_PATH=~/.opentelemetry_genai_rate_limit.db
```

**Note**: The repo uses **comma-separated** emitter names. The entry point name `rate_limit_predictor` is appended to the list:
- Standalone: `OTEL_INSTRUMENTATION_GENAI_EMITTERS="rate_limit_predictor"`
- With spans + metrics: `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric,rate_limit_predictor"`
- With spans + metrics + events: `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"`

### 3.3 End-to-End Testing

**Test Scenario:**
1. Run multi-agent travel planner 5-10 times
2. Verify SQLite database creation and data storage
3. Check workflow pattern learning
4. Verify trace-level aggregation
5. Test prediction accuracy
6. Validate event emission

---

## Phase 4: Demo Preparation (Day 3)

### 4.1 Demo Script

**Create**: `examples/demo_rate_limit_prediction.py`

```python
# Demo script showing:
# 1. Run workflow multiple times
# 2. Show learned patterns
# 3. Show real-time predictions
# 4. Show workflow completion warnings
```

### 4.2 Documentation

**Files to Create/Update:**
- `README.rst` - Package documentation (following repo convention for util packages)
- `CHANGELOG.md` - Package-level changelog (initial entry)
- Update top-level `CHANGELOG.md` - Add entry for new package (if repo convention requires)
- Update `version.py` - Set initial version (following repo versioning scheme)
- Add to main `README.md` - Reference new package in appropriate section

**README Sections:**
- Overview
- Installation
- Configuration
- Usage examples
- Event schema
- Troubleshooting

### 4.3 Demo Video Script

**Key Points to Show:**
1. Problem: Agentic apps hit rate limits unexpectedly
2. Solution: Tokenator predicts breaches before they happen
3. Features:
   - Trace-level aggregation
   - Workflow pattern learning
   - Multi-limit prediction (TPM, weekly, monthly)
   - Early warnings
4. Live demo: Run travel planner, show predictions
5. Results: "You'll hit rate limit in 8 minutes" warning

---

## Phase 5: Polish & Production Readiness (Day 4)

### 5.1 Error Handling

**Add:**
- Graceful degradation if SQLite fails
- Fallback if provider API unavailable
- Logging for debugging
- Exception handling in all critical paths

### 5.2 Performance Optimization

**Optimize:**
- SQLite indexes for fast queries
- Aggregation cache for common windows
- Batch writes where possible
- Connection pooling

### 5.3 Configuration Options

**Environment Variables:**
```bash
# Enable/disable rate limit prediction (default: true if emitter is enabled)
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_ENABLED=true

# Warning threshold (0.0-1.0, default: 0.8 = 80% utilization)
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_WARNING_THRESHOLD=0.8

# SQLite database path (default: ~/.opentelemetry_genai_rate_limit.db)
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_DB_PATH=~/.opentelemetry_genai_rate_limit.db

# Data retention in days (default: 90)
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_RETENTION_DAYS=90

# Enable aggregation cache for performance (default: true)
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_ENABLE_CACHE=true

# EMA smoothing factor for workflow pattern learning (default: 0.3, range: 0.0-1.0)
# Higher values = more reactive to recent runs; lower = smoother, more stable predictions
OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_EMA_ALPHA=0.3
```

### 5.4 Linting & Code Quality

**Tasks:**
- [ ] Run `make lint` and fix issues
- [ ] Add type hints to all functions
- [ ] Add docstrings (Google style)
- [ ] Verify Python 3.10+ compatibility

---

## Implementation Checklist

### Day 1: Foundation
- [ ] Create package structure
- [ ] Set up `pyproject.toml` with entry points
- [ ] Implement SQLite schema (`models.py`)
- [ ] Implement basic tracker (`tracker.py`)
- [ ] Create provider clients (`providers/`)

### Day 2: Core Logic
- [ ] Implement prediction algorithms (`predictor.py`)
- [ ] Implement emitter (`emitter.py`)
- [ ] Add trace/workflow correlation
- [ ] Add workflow pattern learning

### Day 3: Integration
- [ ] Write unit tests
- [ ] Test with multi-agent travel planner
- [ ] Verify event emission
- [ ] Create demo script

### Day 4: Polish
- [ ] Add error handling
- [ ] Optimize performance
- [ ] Write documentation
- [ ] Prepare demo

---

## File-by-File Breakdown

### `pyproject.toml`
```toml
[project]
name = "opentelemetry-util-genai-rate-limit-predictor"
dependencies = [
    "opentelemetry-api>=1.31.0",
    "opentelemetry-util-genai>=0.1.0",
]

[project.entry-points."opentelemetry_util_genai_emitters"]
rate_limit_predictor = "opentelemetry.util.genai.rate_limit.emitter:load_emitters"
```

**Emitter Registration**: The entry point name `rate_limit_predictor` must match a comma-separated token in `OTEL_INSTRUMENTATION_GENAI_EMITTERS`:
- `OTEL_INSTRUMENTATION_GENAI_EMITTERS="rate_limit_predictor"` (standalone)
- `OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"` (combined with spans, metrics, and events)

### `version.py`
```python
__version__ = "0.1.0"
```

### `models.py`
- SQLite schema definitions
- CREATE TABLE statements
- Index definitions

### `tracker.py`
- `SQLiteTokenTracker` class
- Database operations
- Trace aggregation
- Workflow pattern learning

### `providers/base.py`
- `RateLimitProvider` abstract class
- `RateLimits` dataclass

### `providers/openai.py`
- `OpenAIRateLimitProvider` implementation
- Hardcoded free tier limits

### `predictor.py`
- `MultiLimitPredictor` class
- `WorkflowCompletionPredictor` class
- Prediction algorithms

### `emitter.py`
- `RateLimitPredictorEmitter` class
- `load_emitters()` function
- Event emission logic

---

## Success Criteria

1. ✅ Package installs and registers correctly
2. ✅ SQLite database created and populated
3. ✅ Trace-level aggregation works
4. ✅ Workflow patterns learned after 5-10 runs
5. ✅ Predictions generated for TPM, weekly, monthly limits
6. ✅ Workflow completion predictions accurate
7. ✅ Events emitted to OpenTelemetry
8. ✅ Demo runs successfully

---

## Quick Start Commands

```bash
# 1. Create package
mkdir -p util/opentelemetry-util-genai-rate-limit-predictor/{src/opentelemetry/util/genai/rate_limit,tests}

# 2. Install in dev mode
pip install -e ./util/opentelemetry-util-genai-rate-limit-predictor

# 3. Enable emitter (comma-separated list, entry point name must match a token)
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event,rate_limit_predictor"
# Or standalone:
# export OTEL_INSTRUMENTATION_GENAI_EMITTERS="rate_limit_predictor"

# 4. Run test app
python instrumentation-genai/opentelemetry-instrumentation-langchain/examples/multi_agent_travel_planner/main.py

# 5. Check SQLite database (default path)
sqlite3 ~/.opentelemetry_genai_rate_limit.db "SELECT * FROM trace_token_usage;"
```

---

## Architecture Highlights

### Trace-Level Aggregation
- All spans in a trace are aggregated
- Token usage tracked per trace_id
- Enables workflow-level predictions

### Workflow Pattern Learning
- Exponential moving average (EMA) of historical runs
- **Default smoothing factor (α)**: `0.3` — gives ~70% weight to history, ~30% to the latest run. This means the predictor adapts to recent changes but is not overly sensitive to outlier runs.
- Formula: `ema_new = α × latest_tokens + (1 - α) × ema_previous`
- Configurable via env var: `OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_EMA_ALPHA=0.3` (range 0.0–1.0; higher = more reactive to recent runs)
- Learns typical token usage per workflow type
- Improves predictions over time

### Multi-Limit Prediction
- **TPM**: Tokens per minute (immediate concern)
- **Weekly**: Weekly token limits
- **Monthly**: Monthly input/completion limits
- **RPM**: Requests per minute

### Workflow Completion Prediction
- Predicts if entire workflow can complete
- Based on current progress vs. learned patterns
- Warns if rate limits will be hit before completion

---

## Database Schema

### `token_usage`
- Individual span-level token records
- Includes trace_id, span_id for correlation
- Indexed for fast queries

### `trace_token_usage`
- Aggregated token usage per trace
- Tracks workflow name, type, status
- Updated incrementally as spans complete

### `workflow_patterns`
- Learned patterns from completed workflows
- Exponential moving average of token usage
- Used for prediction baseline

---

## Event Schema

### Attribute Namespaces

**Standard OpenTelemetry GenAI Attributes** (from semantic conventions):
- `gen_ai.provider.name` - Provider name (e.g., "openai")
- `gen_ai.request.model` - Model name (e.g., "gpt-4o-mini")
- `gen_ai.workflow.name` - Workflow identifier
- `trace_id` - OpenTelemetry trace ID (hex format)
- `span_id` - OpenTelemetry span ID (hex format)

**Custom Rate Limit Attributes** (vendor-specific extensions):
- `rate_limit.*` - Rate limit prediction attributes
- `workflow.*` - Workflow completion prediction attributes

These custom attributes are clearly namespaced to avoid collisions with standard semantic conventions.

### Rate Limit Warning Event
```json
{
  "event.name": "gen_ai.rate_limit.warning",
  "gen_ai.provider.name": "openai",
  "gen_ai.request.model": "gpt-4o-mini",
  "rate_limit.type": "tpm",
  "rate_limit.current_usage": 180000,
  "rate_limit.limit": 200000,
  "rate_limit.utilization_percent": 90.0,
  "rate_limit.will_breach": false,
  "rate_limit.time_to_breach_seconds": 120,
  "rate_limit.recommendation": "WARNING: Approaching TPM limit..."
}
```

### Workflow Completion Warning Event
```json
{
  "event.name": "gen_ai.workflow.completion.warning",
  "trace_id": "2dfc14dffa52abf855a1fdcd9c52c83a",
  "gen_ai.workflow.name": "multi_agent_travel_planner",
  "workflow.current_tokens": 45230,
  "workflow.predicted_total_tokens": 100000,
  "workflow.progress_percent": 45.2,
  "workflow.tokens_remaining": 54770,
  "workflow.rate_limit_remaining": 12000,
  "workflow.will_hit_rate_limit": true,
  "workflow.can_complete": false,
  "workflow.recommendation": "CRITICAL: Workflow will hit rate limit..."
}
```

**Event Emission**: Events are emitted via OpenTelemetry Logs API (content_logger) in the `content_events` category, consistent with other event emitters in the repository.

---

## Next Steps

1. **Start with Phase 1**: Create package structure
2. **Implement core tracker first**: SQLite foundation is critical
3. **Add prediction logic incrementally**: Test each limit type separately
4. **Test with real app early**: Use multi-agent travel planner
5. **Iterate based on results**: Refine predictions based on actual usage

---

## Known Limitations & Future Enhancements

### Hackathon Limitations (Acceptable for MVP)

1. **Hardcoded Rate Limits**: Free tier limits are hardcoded (as of 2026-02-06, sourced from OpenAI platform dashboard). These values may change over time and differ by account tier. Production should:
   - Fetch from provider management APIs (e.g., OpenAI `/organization/limits`)
   - Support environment variable overrides (e.g., `OTEL_INSTRUMENTATION_GENAI_RATE_LIMIT_OPENAI_TPM=200000`)
   - Allow per-model/account-tier configuration

2. **Multi-Process Concurrency**: SQLite tracks token usage per-process only. Token usage from separate processes (e.g., eval workers) is not aggregated. This is acceptable as token usage is primarily tracked within span/trace context.

3. **Schema Migrations**: No migration strategy for schema evolution. If schema changes are needed, users should delete the database file (`~/.opentelemetry_genai_rate_limit.db`) to recreate with the new schema. Production version should include proper versioning and migration support (e.g., schema version table + incremental ALTER statements).

4. **Synchronous Prediction**: Predictions run synchronously in `on_end()`. For lightweight predictions (SQLite queries + simple calculations) this is fine, but if predictions become expensive, consider:
   - Async/background processing
   - Sampling (only predict on a subset of invocations)
   - Caching recent predictions with a short TTL

5. **Rolling Windows Only**: TPM/weekly/monthly windows are rolling (last N seconds), not calendar-aligned. Calendar-aligned windows can be added later if needed.

6. **Emitter Configuration**: The emitter uses the standard comma-separated `OTEL_INSTRUMENTATION_GENAI_EMITTERS` format. A small integration test should validate that `rate_limit_predictor` loads correctly alongside other emitters (e.g., `span_metric_event,rate_limit_predictor`).

### Future Enhancements

- Provider API integration for dynamic rate limit fetching
- Configuration file support for custom limits
- Schema versioning and migration support
- Multi-process aggregation via shared storage (Redis, etc.)
- Async prediction processing
- Calendar-aligned time windows (per-month, per-week boundaries)
- ML-based prediction models for improved accuracy

---

## References

- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Splunk OTel Python Contrib Architecture](README.packages.architecture.md)
- [Agent Instructions](AGENTS.md)

---

**Status**: Planning Phase  
**Last Updated**: 2026-02-06  
**Owner**: Hackathon Team
