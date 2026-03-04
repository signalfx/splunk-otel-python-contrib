## ADDED Requirements

### Requirement: All instrumentation-genai packages SHALL depend on semconv >= 0.60b1

Every package under `instrumentation-genai/` SHALL declare `opentelemetry-semantic-conventions >= 0.60b1` in its `pyproject.toml` dependencies section.

#### Scenario: Langchain package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-langchain/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: CrewAI package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-crewai/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: LlamaIndex package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-llamaindex/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: OpenAI v2 package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-openai-v2/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: OpenAI Agents v2 package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-openai-agents-v2/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: AI Defense package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-aidefense/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: Weaviate package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-weaviate/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: FastMCP package dependency
- **WHEN** a user inspects `instrumentation-genai/opentelemetry-instrumentation-fastmcp/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

### Requirement: All util packages SHALL depend on semconv >= 0.60b1

Every package under `util/` that depends on `opentelemetry-semantic-conventions` SHALL declare `>= 0.60b1` in its `pyproject.toml` dependencies section.

#### Scenario: Util genai core dependency
- **WHEN** a user inspects `util/opentelemetry-util-genai/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: Emitters splunk dependency
- **WHEN** a user inspects `util/opentelemetry-util-genai-emitters-splunk/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: Openlit translator dependency
- **WHEN** a user inspects `util/opentelemetry-util-genai-openlit-translator/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: Langsmith translator dependency
- **WHEN** a user inspects `util/opentelemetry-util-genai-langsmith-translator/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

#### Scenario: Traceloop translator dependency
- **WHEN** a user inspects `util/opentelemetry-util-genai-traceloop-translator/pyproject.toml`
- **THEN** the dependencies list SHALL contain `"opentelemetry-semantic-conventions >= 0.60b1"`

### Requirement: Package versions SHALL be bumped

Each affected package SHALL have its minor version incremented in `version.py` to reflect the dependency change.

#### Scenario: Version bump is applied
- **WHEN** the semconv dependency is upgraded in a package's `pyproject.toml`
- **THEN** the corresponding `version.py` SHALL have its minor version incremented (e.g., `0.5.0` → `0.6.0`)

### Requirement: Compatible installation with splunk-otel-python

The upgraded packages SHALL be installable alongside `splunk-otel-python >= 2.9.0` without pip dependency conflicts.

#### Scenario: No conflict with splunk distro
- **WHEN** a user runs `pip install splunk-opentelemetry[all] opentelemetry-instrumentation-langchain`
- **THEN** pip SHALL resolve all dependencies without conflicts related to `opentelemetry-semantic-conventions`

### Requirement: CHANGELOG entries SHALL be added

Each affected package's CHANGELOG.md (or the root CHANGELOG.md) SHALL document the semconv version upgrade.

#### Scenario: Changelog documents the upgrade
- **WHEN** a developer reviews the CHANGELOG
- **THEN** there SHALL be an entry noting the `opentelemetry-semantic-conventions` dependency upgrade to `>= 0.60b1`
