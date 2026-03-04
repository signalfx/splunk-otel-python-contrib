## 1. Upgrade util packages (dependency order: these are depended on by instrumentation packages)

- [x] 1.1 Update `util/opentelemetry-util-genai/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.57b0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 1.2 Update `util/opentelemetry-util-genai-emitters-splunk/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.57b0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 1.3 Update `util/opentelemetry-util-genai-openlit-translator/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.52b1` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 1.4 Update `util/opentelemetry-util-genai-langsmith-translator/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.52b1` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 1.5 Update `util/opentelemetry-util-genai-traceloop-translator/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.52b1` to `opentelemetry-semantic-conventions >= 0.60b1`

## 2. Upgrade instrumentation-genai packages

- [x] 2.1 Update `instrumentation-genai/opentelemetry-instrumentation-openai-v2/pyproject.toml`: change `opentelemetry-semantic-conventions >= 0.58b0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.2 Update `instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/pyproject.toml`: change `opentelemetry-semantic-conventions >= 0.58b0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.3 Update `instrumentation-genai/opentelemetry-instrumentation-langchain/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.4 Update `instrumentation-genai/opentelemetry-instrumentation-crewai/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.5 Update `instrumentation-genai/opentelemetry-instrumentation-llamaindex/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.6 Update `instrumentation-genai/opentelemetry-instrumentation-aidefense/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.7 Update `instrumentation-genai/opentelemetry-instrumentation-weaviate/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`
- [x] 2.8 Update `instrumentation-genai/opentelemetry-instrumentation-fastmcp/pyproject.toml`: change `opentelemetry-semantic-conventions ~= 0.59b0.dev0` to `opentelemetry-semantic-conventions >= 0.60b1`

## 3. Bump package versions

- [x] 3.1 Bump minor version in `util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py`
- [x] 3.2 Bump minor version in `util/opentelemetry-util-genai-emitters-splunk/src/opentelemetry/util/genai/emitters/splunk/version.py`
- [x] 3.3 Bump minor version in `util/opentelemetry-util-genai-openlit-translator/src/opentelemetry/util/genai/openlit_translator/version.py`
- [x] 3.4 Bump minor version in `util/opentelemetry-util-genai-langsmith-translator/src/opentelemetry/util/genai/langsmith_translator/version.py`
- [x] 3.5 Bump minor version in `util/opentelemetry-util-genai-traceloop-translator/src/opentelemetry/util/genai/traceloop_translator/version.py`
- [x] 3.6 Bump minor version in each instrumentation-genai package's `version.py`

## 4. Documentation and changelog

- [x] 4.1 Add CHANGELOG.md entry documenting the semconv upgrade to `>= 0.60b1` for all affected packages

## 5. Validation

- [x] 5.1 Run `make lint` to verify no formatting or linting issues (all errors are pre-existing in untracked files, none from our changes)
- [x] 5.2 Run tests for util packages: `pytest ./util/opentelemetry-util-genai/tests/ -v` (99 passed)
- [x] 5.3 Run tests for instrumentation packages (langchain: 12 passed; crewai: 51 passed, 8 pre-existing failures)
- [x] 5.4 Verify `pip install` resolves cleanly with no semconv version conflicts (semconv 0.60b1 installed successfully)
