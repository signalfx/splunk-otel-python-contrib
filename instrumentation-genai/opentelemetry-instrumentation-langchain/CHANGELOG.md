# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.8

### Added
- **Error classification for interrupts** — `GraphInterrupt`, `NodeInterrupt`, and `Interrupt` exceptions are now classified as interrupts (not errors), leaving span status as `UNSET` (default) instead of `StatusCode.ERROR`. `CancelledError` and `TaskCancelledError` are classified as cancellations.
- **Resume detection** — `gen_ai.workflow.command = "resume"` set on root workflow span when resuming via `Command(resume=...)` or from a LangGraph checkpoint with `checkpoint_id`.
- **Orphan span guard** — Prevents orphan root spans when LangGraph replays interrupted nodes during resume.
- **Command input handling** — LangGraph passes a `Command` object (not a dict) as `inputs` on resume; the callback handler now normalises this without crashing and captures the `resume` value as a user input message on the workflow span. Dict resume values are JSON-serialized; non-resume commands use the string representation.
- **LangGraph node name fallback** — When `serialized` is `None` (LangGraph ≥1.0), `langgraph_node` from callback metadata is used as the step name.

## Version 0.1.7 - 2026-03-06

### Added
- **Automatic `gen_ai.conversation.id` inference from LangGraph `thread_id`** — When LangGraph's `configurable.thread_id` is present in callback metadata, it is automatically mapped to `gen_ai.conversation.id` on root Workflow and AgentInvocation spans. No manual `genai_context()` wrapping needed. Explicit `genai_context(conversation_id=...)` always takes priority over the inferred value. Also recognizes `conversation_id` key in metadata (checked before `thread_id`).

## Version 0.1.7 - 2026-01-28

### Added
- **Rate limiter test example** - New `rate-limiter-test` example demonstrating evaluation rate limiting configuration and behavior

## Version 0.1.6 - 2026-01-28

### Changed
- **Fixed input and output messages** - Workflow and AgentInvocation's input and output messages are serialized per item.

## Version 0.1.5 - 2026-01-27

### Changed
- **Use structured message fields** - Workflow and AgentInvocation now use `input_messages`/`output_messages` structured fields instead of legacy string fields for proper JSON serialization.
- **Improved JSON serialization** - `_serialize()` helper now uses `json.dumps(obj, default=str)` for reliable JSON output with non-serializable objects.
- **Removed input/output from Step** - Step spans no longer capture redundant input/output data.

### Fixed
- **Agent/workflow message rendering** - Fixed issue where agent and workflow `gen_ai.input.messages` were rendered as Python repr strings instead of valid JSON.

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-instrumentation-langchain
