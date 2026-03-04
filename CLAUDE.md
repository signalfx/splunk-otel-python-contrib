# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Splunk OpenTelemetry Python Contrib - A GenAI instrumentation framework with pluggable emitters and evaluation support. The core abstraction separates *instrumentation capture* (GenAI Types) from *telemetry emission* (pluggable Emitters), enabling vendor-specific enrichments without modifying instrumentation code.

**Status**: Alpha preview - may include breaking changes.

## Build & Development Commands

### Environment Setup
```bash
# Create venv and install tools
python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install pre-commit -c dev-requirements.txt && pre-commit install

# Install GenAI packages (editable, no deps)
pip install -e util/opentelemetry-util-genai --no-deps
pip install -e util/opentelemetry-util-genai-evals --no-deps
pip install -e util/opentelemetry-util-genai-evals-deepeval --no-deps
pip install -e util/opentelemetry-util-genai-emitters-splunk --no-deps
pip install -e instrumentation-genai/opentelemetry-instrumentation-langchain --no-deps
pip install -r dev-genai-requirements.txt

# Required environment variables
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

### Testing
```bash
# Run tests for specific package
pytest util/opentelemetry-util-genai/tests/ -v
pytest instrumentation-genai/opentelemetry-instrumentation-langchain/tests/ -v

# Run single test file
pytest util/opentelemetry-util-genai/tests/test_metrics.py -v

# With coverage
pytest util/opentelemetry-util-genai/tests/ --cov=opentelemetry.util.genai
```

### Linting & Formatting
```bash
# Fix all issues (recommended)
make lint

# Or using ruff directly
ruff check --fix .
ruff format .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Building Packages
```bash
cd util/opentelemetry-util-genai
hatch build
```

## Architecture

### Core Flow
```
Instrumentation → GenAI Types → TelemetryHandler → CompositeEmitter → OpenTelemetry SDK
                                      ↓
                              Completion Callbacks (Evaluation Manager)
                                      ↓
                              EvaluationResult → Emitters
```

### Key Components

**`util/opentelemetry-util-genai/`** - Core framework:
- `types.py` - Data model: `LLMInvocation`, `AgentInvocation`, `Workflow`, `EvaluationResult`, etc.
- `handler.py` - `TelemetryHandler` with lifecycle methods (`start_llm`, `stop_llm`, `fail_llm`)
- `emitters/composite.py` - Fan-out dispatcher with ordered category dispatch
- `emitters/configuration.py` - Emitter directive parsing from env vars
- `config.py` - Environment variable configuration

**`instrumentation-genai/`** - Framework instrumentations:
- `opentelemetry-instrumentation-langchain/` - LangChain
- `opentelemetry-instrumentation-crewai/` - CrewAI
- `opentelemetry-instrumentation-openai-v2/` - OpenAI
- `opentelemetry-instrumentation-llamaindex/` - LlamaIndex

**`util/opentelemetry-util-genai-evals*/`** - Evaluation framework and DeepEval integration

### Emitter Categories & Ordering
- **Start order**: `span` → `metrics` → `content_events`
- **End order**: `evaluation` → `metrics` → `content_events` → `span`

### Plugin System
Entry point groups for extensibility:
- `opentelemetry_util_genai_emitters` - Custom emitters
- `opentelemetry_util_genai_evaluators` - Custom evaluators
- `opentelemetry_util_genai_completion_callbacks` - Completion callbacks

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitter selection: `span`, `span_metric`, `span_metric_event`, plus extras |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS_<CATEGORY>` | Category overrides with directives: `append:`, `prepend:`, `replace:` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Enable message capture (default: disabled) |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | Evaluator config: `Deepeval(LLMInvocation(bias,toxicity))` |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` | Trace-id ratio sampling (0-1, default: 1.0) |

## Code Quality

- **Linter**: Ruff v0.14.1 (line length: 79)
- **Type checker**: Pyright (strict mode, Python 3.9+)
- **Pre-commit hooks**: ruff lint/format, uv-lock, rstcheck
- Commits require signing and [Splunk CLA](https://github.com/splunk/cla-agreement/blob/main/CLA.md)
