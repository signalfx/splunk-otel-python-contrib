## Context

The Splunk OpenTelemetry Python Contrib repository contains 13 packages (8 instrumentation-genai + 5 util) that depend on `opentelemetry-semantic-conventions`. These packages currently pin to versions ranging from `~= 0.52b1` to `~= 0.59b0.dev0`. The upstream Splunk distribution (`splunk-otel-python`) has moved to `opentelemetry-semantic-conventions==0.60b1`, creating pip dependency resolution conflicts for end users.

Current state by package group:

| Group | Packages | Current Constraint |
|-------|----------|--------------------|
| Util (translators) | openlit-translator, langsmith-translator, traceloop-translator | `~= 0.52b1` |
| Util (core) | util-genai, emitters-splunk | `~= 0.57b0` |
| Instrumentation (OpenAI) | openai-v2, openai-agents-v2 | `>= 0.58b0` |
| Instrumentation (rest) | langchain, aidefense, weaviate, llamaindex, crewai, fastmcp | `~= 0.59b0.dev0` |

## Goals / Non-Goals

**Goals:**

- Align all packages to `opentelemetry-semantic-conventions >= 0.60b1` so they are compatible with the latest `splunk-otel-python` distribution.
- Use `>= 0.60b1` (not `~=` or `==`) to allow forward-compatible resolution with future semconv releases.
- Bump each affected package's minor version to signal the dependency change.
- Remove the need for users to pin `splunk-opentelemetry[all]==2.8.0` as a workaround.

**Non-Goals:**

- Adopting new semantic convention attributes introduced in 0.60b1 — that is a separate follow-up change.
- Changing the `opentelemetry-api` or `opentelemetry-sdk` version constraints.
- Modifying any instrumentation logic or emitter behavior.

## Decisions

### D1: Use `>= 0.60b1` constraint style

**Decision**: Use `>= 0.60b1` rather than `~= 0.60b1` or `== 0.60b1`.

**Rationale**: The `>=` operator allows users to install any future compatible version without requiring a release from this repo for every upstream semconv bump. The `~=` operator is overly restrictive for a package that is backward-compatible within the 0.x beta series. The OpenAI packages already use `>=` style successfully.

**Alternatives considered**:
- `~= 0.60b1` — Too restrictive; would block `0.61b0` and above.
- `== 0.60b1` — Pins exactly; would require constant releases to track upstream.

### D2: Bump minor version for all affected packages

**Decision**: Increment the minor version in each package's `version.py` (e.g., `0.5.0` → `0.6.0`).

**Rationale**: A dependency floor change is a meaningful compatibility change that consumers should be aware of. Minor version bump signals this without implying breaking API changes.

### D3: Single batch change across all packages

**Decision**: Apply the version bump to all 13 packages in a single PR.

**Rationale**: The packages are released together and the root cause (incompatibility with splunk-otel-python) affects all of them. Splitting into multiple PRs adds coordination overhead with no benefit.

## Risks / Trade-offs

- **[Risk] Packages using features removed in 0.60b1** → Low probability. Semconv beta releases are additive. Mitigation: run full test suite post-change.
- **[Risk] Users on older splunk-otel-python versions** → Users on `splunk-opentelemetry < 2.9.0` may get a newer semconv than expected. Mitigation: `>= 0.60b1` is a floor, not a ceiling; pip will resolve the highest compatible version across all constraints.
- **[Risk] CI uses dev/pre-release semconv** → Some packages currently reference `0.59b0.dev0` (a dev release). Moving to `0.60b1` (a beta release) is more stable. No risk.
