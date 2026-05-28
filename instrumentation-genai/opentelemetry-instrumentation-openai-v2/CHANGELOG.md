# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 0.1.2 (2026-05-19)

### Fixed

- Fix `AttributeError: 'StreamWrapper' object has no attribute 'headers'` when
  using `with_raw_response.create(stream=True)` (e.g. via LiteLLM's Azure provider).
  `_parse_response` was calling `.parse()` on the `LegacyAPIResponse` before wrapping
  in `StreamWrapper`, discarding the raw HTTP headers. `StreamWrapper` now captures
  headers from the `LegacyAPIResponse` before it is parsed and exposes them directly,
  and adds a `parse()` method returning `self` so callers can treat the wrapper as
  a drop-in for the raw response. Also adds `__getattr__` to proxy any other unknown
  attributes to the underlying stream. Inspired by upstream fix
  ([opentelemetry-python-contrib#4184](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4184),
  fixes [#4113](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/4113)).

### Added

- Add `gen_ai.tool.definitions` attribute on LLM spans when
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` and
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_TOOL_DEFINITIONS=true`
- Add `gen_ai.request.stream` attribute for streaming requests
- Add `gen_ai.response.time_to_first_chunk` attribute and metric for streaming requests

### Fixed

- Fix PyPI badge, install command, and references in README.rst to use correct
  `splunk-otel-instrumentation-openai` package name instead of upstream
- Fix project URLs in pyproject.toml to point to SDOT repo (`signalfx/splunk-otel-python-contrib`)

### Changed

- **Always populate messages and tool arguments on Python objects** — `input_messages`, `output_messages`, and tool call `arguments` are now always set on `LLMInvocation`/`ToolCall` regardless of `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. The emitter layer controls what reaches telemetry, enabling evaluators to access full content even in `NO_CONTENT` mode.

## Version 0.1.0 (2026-02-05)

Initial release of `splunk-otel-instrumentation-openai` package.

- Fix `AttributeError` when handling `LegacyAPIResponse` (from `with_raw_response`)
  ([#4017](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4017))
- Add support for chat completions choice count and stop sequences span attributes
  ([#4028](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4028))
- Migrate chat completions to genai-util TelemetryHandler
  ([#106](https://github.com/signalfx/splunk-otel-python-contrib/pull/106))
- Migrate embeddings to genai-util TelemetryHandler
  ([#114](https://github.com/signalfx/splunk-otel-python-contrib/pull/114))
- Update tool call handling
  ([#135](https://github.com/signalfx/splunk-otel-python-contrib/pull/135))
- Add suppression key handling
  ([#155](https://github.com/signalfx/splunk-otel-python-contrib/pull/155))
- Move events/logs and metrics to handler-based emitters
  ([#158](https://github.com/signalfx/splunk-otel-python-contrib/pull/158))
- Fix service tier attribute names: use `GEN_AI_OPENAI_REQUEST_SERVICE_TIER` for request
  attributes and `GEN_AI_OPENAI_RESPONSE_SERVICE_TIER` for response attributes
  ([#3920](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3920))
- Added support for OpenAI embeddings instrumentation
  ([#3461](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3461))
- Record prompt and completion events regardless of span sampling decision
  ([#3226](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3226))
- Filter out attributes with the value of NotGiven instances
  ([#3760](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3760))
- Migrate off the deprecated events API to use the logs API
  ([#3625](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3628))
- Coerce openai response_format to semconv format
  ([#3073](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3073))
- Add example to `opentelemetry-instrumentation-openai-v2`
  ([#3006](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3006))
- Support for `AsyncOpenAI/AsyncCompletions`
  ([#2984](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2984))
- Add metrics
  ([#3180](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3180))
- Use generic `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable
  to control if content of prompt, completion, and other messages is captured
  ([#2947](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2947))
- Update OpenAI instrumentation to Semantic Conventions v1.28.0: add new attributes
  and switch prompts and completions to log-based events
  ([#2925](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2925))
- Initial OpenAI instrumentation
  ([#2759](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/2759))
