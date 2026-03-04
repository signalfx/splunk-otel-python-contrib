## Why

The `opentelemetry-semantic-conventions` dependency is pinned to various outdated versions across the instrumentation-genai and util packages (ranging from `~= 0.52b1` to `~= 0.59b0.dev0`). The upstream `splunk-otel-python` distribution has moved to `0.60b1`, causing dependency resolution conflicts when users install the Splunk distro alongside these instrumentation packages. The temporary workaround of pinning `splunk-opentelemetry[all]==2.8.0` is fragile and blocks adoption of newer distro releases.

## What Changes

- Upgrade `opentelemetry-semantic-conventions` version constraint to `>= 0.60b1` in all instrumentation-genai packages:
  - `opentelemetry-instrumentation-openai-v2` (currently `>= 0.58b0`)
  - `opentelemetry-instrumentation-openai-agents-v2` (currently `>= 0.58b0`)
  - `opentelemetry-instrumentation-langchain` (currently `~= 0.59b0.dev0`)
  - `opentelemetry-instrumentation-aidefense` (currently `~= 0.59b0.dev0`)
  - `opentelemetry-instrumentation-weaviate` (currently `~= 0.59b0.dev0`)
  - `opentelemetry-instrumentation-llamaindex` (currently `~= 0.59b0.dev0`)
  - `opentelemetry-instrumentation-crewai` (currently `~= 0.59b0.dev0`)
  - `opentelemetry-instrumentation-fastmcp` (currently `~= 0.59b0.dev0`)
- Upgrade `opentelemetry-semantic-conventions` version constraint in util packages:
  - `opentelemetry-util-genai` (currently `~= 0.57b0`)
  - `opentelemetry-util-genai-emitters-splunk` (currently `~= 0.57b0`)
  - `opentelemetry-util-genai-openlit-translator` (currently `~= 0.52b1`)
  - `opentelemetry-util-genai-langsmith-translator` (currently `~= 0.52b1`)
  - `opentelemetry-util-genai-traceloop-translator` (currently `~= 0.52b1`)
- Bump minor version in each affected package's `version.py`.
- Update CHANGELOG.md entries for all affected packages.

## Capabilities

### New Capabilities

- `semconv-version-upgrade`: Align all packages to `opentelemetry-semantic-conventions >= 0.60b1` to resolve dependency conflicts with `splunk-otel-python` and enable compatibility with the latest Splunk distro releases.

### Modified Capabilities

(none — no spec-level behavior changes, this is a dependency version bump)

## Impact

- **Dependencies**: All 13 packages listed above will have their `pyproject.toml` updated.
- **Compatibility**: Resolves pip dependency conflicts when installing alongside `splunk-otel-python >= 2.9.0` (which requires semconv `0.60b1`). Users will no longer need to pin `splunk-opentelemetry[all]==2.8.0` as a workaround.
- **Risk**: Low — semantic conventions are backward-compatible within the 0.x beta series. Any new attributes/constants are additive. Existing code referencing semconv attributes will continue to work.
- **Testing**: Existing test suites should pass without modification. CI lint and pytest across Python 3.10-3.13 must be green.
