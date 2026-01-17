# Agent Instructions

> This file provides context for AI coding agents (OpenAI Codex, GitHub Copilot, Cursor, Windsurf, Claude).

## Project Overview

**Splunk OpenTelemetry Python Contrib** - GenAI instrumentation packages for Python providing AI/ML observability through OpenTelemetry.

**Status**: Alpha preview - breaking changes expected.

**Core Purpose of Splunk Distro for OpenTelemetry(SDOT)**: Provides util libraries to separate *instrumentation capture* from *telemetry emission* so:

- Instrumentation authors create neutral GenAI data objects once
- Pluggable emitters produce different telemetry flavors (spans, metrics, events)
- Evaluations (LLM-as-a-judge) run asynchronously through the same pipeline
- Configuration is environment-variable driven

## Repository Structure

```text
├── util/                                    # Core utility packages
│   ├── opentelemetry-util-genai/           # Core: TelemetryHandler, emitters, types
│   ├── opentelemetry-util-genai-evals/     # Async evaluation manager & registry
│   ├── opentelemetry-util-genai-evals-deepeval/  # Deepeval metrics integration
│   ├── opentelemetry-util-genai-emitters-splunk/ # Splunk-specific emitters
│   └── opentelemetry-util-genai-traceloop-translator/ # Traceloop span translation
│
├── instrumentation-genai/                   # Framework instrumentations
│   ├── opentelemetry-instrumentation-langchain/   # LangChain/LangGraph
│   ├── opentelemetry-instrumentation-crewai/      # CrewAI
│   ├── opentelemetry-instrumentation-openai-v2/   # OpenAI SDK
│   ├── opentelemetry-instrumentation-openai-agents-v2/ # OpenAI Agents
│   ├── opentelemetry-instrumentation-llamaindex/  # LlamaIndex
│   └── opentelemetry-instrumentation-aidefense/   # AI Defense
```

## Quick Reference

### Development Commands

```bash
# Setup virtual environment (macOS)
python -m venv .venv && source .venv/bin/activate

# Install dev dependencies
pip install -r dev-requirements.txt
pip install -r dev-genai-requirements.txt

# Install a package in editable mode
pip install -e ./util/opentelemetry-util-genai
pip install -e "./instrumentation-genai/opentelemetry-instrumentation-langchain[instruments,test]"

# Run tests for a specific package
pytest ./util/opentelemetry-util-genai/tests/ -v
pytest ./instrumentation-genai/opentelemetry-instrumentation-langchain/tests/ -v

# Run linting (REQUIRED before commits)
make lint
# Or manually:
ruff check --fix . && ruff format .
```

### CI Checks (must pass)

1. **Lint** (`ci-lint.yaml`): `ruff check .` and `ruff format --check .`
2. **Tests** (`ci-main.yaml`): pytest across Python 3.10-3.13 on Linux/macOS/Windows

## Code Patterns

### Key Types (util/opentelemetry-util-genai/src/opentelemetry/util/genai/)

| File | Purpose |
|------|---------|
| `types.py` | Core dataclasses: `GenAI`, `LLMInvocation`, `AgentInvocation`, `Workflow`, `ToolCall`, `EvaluationResult` |
| `handler.py` | `TelemetryHandler` - lifecycle facade (`start_llm`, `stop_llm`, `fail_llm`, etc.) |
| `interfaces.py` | `EmitterProtocol`, `CompletionCallback`, `Evaluator` protocols |
| `emitters/composite.py` | `CompositeEmitter` - chains emitters by category |
| `emitters/span.py` | Semantic convention span emitter |
| `emitters/metrics.py` | Metrics emitter (duration, tokens) |
| `attributes.py` | Semantic attribute extraction |

### Emitter Protocol

```python
class EmitterProtocol(Protocol):
    def on_start(self, obj: GenAI) -> None: ...
    def on_end(self, obj: GenAI) -> None: ...
    def on_error(self, error: Exception, obj: GenAI) -> None: ...
    def on_evaluation_results(self, results: list[EvaluationResult], obj: GenAI | None) -> None: ...
```

### Plugin Registration (pyproject.toml)

```toml
[project.entry-points.opentelemetry_util_genai_emitters]
my_emitter = "my_package:load_emitters"

[project.entry-points.opentelemetry_util_genai_evaluators]
my_evaluator = "my_package:register"
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OTEL_INSTRUMENTATION_GENAI_ENABLE` | Enable/disable instrumentation | `true` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture message content | `false` |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` | `SPAN`, `EVENT`, `SPAN_AND_EVENT` | `SPAN` |
| `OTEL_INSTRUMENTATION_GENAI_EMITTERS` | Emitter selection | `span` |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | Evaluator configuration | "default value" |
| `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` | Evaluation sampling (0.0-1.0) | `1.0` |

## Code Style

- **Python**: 3.10+ required, type hints for all public APIs
- **Linting**: Ruff (config in `pyproject.toml`)
- **Docstrings**: Google style
- **Dataclasses**: Use for structured data (see `types.py`)
- **Protocols**: Use `typing.Protocol` for interfaces
- **Async**: Use `async/await` for I/O operations

## Testing

- Tests in `tests/` subdirectory of each package and requires this package installed in development mode locally, like ```bash pip install -e ./util/opentelemetry-util-genai```
- Use pytest with fixtures
- Mock external services (LLM providers, etternal APIs, etc.). Do not mock instrumented frameworks unless absolutely necessary.

```bash
# Run with coverage
pytest ./util/opentelemetry-util-genai/tests/ -v --cov=opentelemetry.util.genai
```

## Documentation References

For detailed information, see these files in the repository:

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Core concepts, emitter architecture, evaluation system |
| [README.packages.architecture.md](README.packages.architecture.md) | Package architecture, interfaces, lifecycle diagrams |
| [README.eval-monitoring.md](README.eval-monitoring.md) | Evaluation monitoring metrics plan |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines, PR process |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Detailed development setup for macOS |

## Common Tasks

### Adding a New Instrumentation

1. Create `instrumentation-genai/opentelemetry-instrumentation-{name}/`
2. Copy structure from existing instrumentation (e.g., langchain)
3. Implement a target instrumented framework demo apps, i.e. `instrumentation-genai/opentelemetry-instrumentation-langchain/examples/sre_incident_copilot`
4. Implement callback handler/wrappers for the instrumented functions
5. Don't create telemetry in the instrumentation library, use `util/opentelemetry-util-genai` APIs to create GenAI Types and for lifecycle
6. Validate functionality with real app
7. Add tests with mocked LLM Provider call (test all the way through the framework, don't mock the framework)

### Adding a New Emitter

1. Create class implementing `EmitterProtocol`
2. Add `load_emitters()` function returning `list[EmitterSpec]`
3. Register via entry point in `pyproject.toml`
4. Document configuration environment variables

### Adding a New Evaluator

1. Create class implementing `Evaluator` protocol in a new package (see `util/opentelemetry-util-genai-evals-deepeval` for example)
2. Add `register()` function for entry point discovery
3. Handle evaluation results via `TelemetryHandler.evaluation_results()`

## Debugging

VS Code launch configurations are in `.vscode/launch.json` for debugging examples.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "LangChain Demo S0 Current (span_metric_event)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/instrumentation-genai/opentelemetry-instrumentation-langchain/examples/manual/main.py",
      "python": "${workspaceFolder}/.venv/bin/python",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "DELTA",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED": "true",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment=o11y-for-ai-dev-sergey,scenario=current",
        "OTEL_SERVICE_NAME": "demo-app-util-langchain-dev-sergey",
        "OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE": "SPAN_AND_EVENT",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": "span_metric_event",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS": "deepeval(LLMInvocation(bias,toxicity),AgentInvocation(hallucination))"
      }
    }
  ]
}
```

## Notes for AI Agents

- Always run `make lint` and `make ruff` before committing
- Ask user to provide a file to export OPENAI_API_KEY or other credentials to run the demo app to validate
- Check existing patterns in similar packages before implementing
- Tests are required for all new features
- Environment variables should have `OTEL_INSTRUMENTATION_GENAI_` prefix
- Use semantic conventions from OpenTelemetry GenAI spec where applicable

## Common Pitfalls to Avoid

- Do not try to mock libraries if import in the current env fail. If in doubts - clearly to communicate to user the problem
- always refer to README.md and README.AR
