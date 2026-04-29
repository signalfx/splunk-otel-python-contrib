# Changelog

## Version 0.1.4 - 2026-04-29

### Fixed
- Emit metrics for agent/tool/workflow spans via `handler.finish()` (previously only LLM spans emitted metrics).
- Fixed negative duration on `gen_ai.client.operation.duration` histogram by using monotonic clock offsets.
- Fixed `AgentInvocation`/`AgentCreation`/`Workflow` invocation builders to use current `input_messages`/`output_messages` fields.
- Added `langsmith.trace.session_id` → `gen_ai.conversation.id` mapping (the actual attribute LangSmith OTELExporter uses).
- Added `langsmith.tool.name` → `gen_ai.tool.call.name` mapping.

### Added
- `ToolCall` invocation creation for `execute_tool` spans with tool name, arguments, result, and parent agent context.
- Agent name propagation from parent agent spans to child LLM/tool spans (`gen_ai.agent.name`, `gen_ai.agent.id`).
- Message reconstruction gating now includes `execute_tool` spans.
- Streaming attribute mapping: `langsmith.request.streaming` → `gen_ai.request.stream`.

### Changed
- Bumped `opentelemetry-instrumentation` and `opentelemetry-semantic-conventions` to `~= 0.57b0`.
- Bumped `splunk-otel-util-genai` dependency to `>=0.1.8`.

## Version 0.1.3 - 2026-04-29
- Fixed namespace package conflicts between translator packages.
- Added CI tests.

## Version 0.1.2 - 2026-04-10
- Updated requests version and python version.

## Version 0.1.1 - 2026-02-06
- Resolved issue with missing Agent evaluations.
- Removed redundant calls which was causing duplicate metrics & evaluations.

## [0.1.0] - 2025-01-29

### Added
- Initial release of Langsmith to GenAI semantic convention translator
- `LangsmithSpanProcessor` for translating Langsmith span attributes to GenAI semantic conventions
- Auto-enable functionality on package installation
- Support for attribute transformations:
  - Content & Messages (entity input/output to gen_ai.input/output.messages)
  - Model & System attributes (ls_provider, ls_model_name)
  - Request parameters (temperature, max_tokens, top_p, top_k)
  - Token usage metrics
  - Tool/function calling attributes
  - Session and run tracking
  - Agent and workflow attributes
- Message reconstruction from LangChain serialization formats
- Content normalization for various provider formats (OpenAI, Anthropic, Google)
- Support for conditional transformation rules
- Environment variable `OTEL_INSTRUMENTATION_GENAI_LANGSMITH_DISABLE` to disable auto-registration
