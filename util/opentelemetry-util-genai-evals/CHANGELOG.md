# Changelog

All notable changes to this repository are documented in this file.

## Version 0.1.8 - 2026-02-06

### Fixed
- **Logging visibility** - INFO level log messages from the evals bootstrap, proxy, manager, and worker modules are now always visible (e.g., "Using separate process evaluation mode"). DEBUG level messages require `OTEL_INSTRUMENTATION_GENAI_DEBUG=true`.

## Version 0.1.7 - 2026-01-28

### Added

- **Evaluation Rate Limiting** - New `EvaluationAdmissionController` with token bucket rate limiter for controlling evaluation throughput
  - `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_ENABLE` - Enable/disable rate limiting (default: true)
  - `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_RPS` - Requests per second limit (default: 10)
  - `OTEL_INSTRUMENTATION_GENAI_EVALUATION_RATE_LIMIT_BURST` - Burst capacity (default: 20)
  - Thread-safe token bucket implementation prevents evaluation queue overload

- **Separate Process Evaluation Mode** - Run evaluations in an isolated child process
  - Prevents DeepEval/OpenAI instrumentation from polluting application telemetry
  - Enable via `OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS=true`
  - Child process runs with `OTEL_SDK_DISABLED=true` for complete isolation
  - Bi-directional IPC using `multiprocessing.Pipe` for invocation/result exchange
  - Automatic fallback to in-process mode if multiprocessing fails

- **New Components**
  - `EvalManagerProxy` - CompletionCallback implementation for parent process
  - `EvalWorker` - Worker class running evaluation loop in child process
  - `NoOpTelemetryHandler` - Stub handler for child process (no telemetry emission)
  - Serialization utilities for GenAI types and EvaluationResult

- **New Environment Variables**
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS` - Enable separate process mode (default: `false`)
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKER_TIMEOUT` - Worker startup timeout (default: 30s)
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULT_TIMEOUT` - Result wait timeout (default: 60s)

- **New Exports from Package**
  - `EvalManagerProxy`
  - `is_separate_process_enabled()`
  - `OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS`
  - `ErrorEvent`
  - `ErrorTracker`

### Changed
- `create_evaluation_manager()` now returns `EvalManagerProxy` when separate process mode is enabled

## Version 0.1.6 - 2026-01-27

- Release 0.1.6 of splunk-otel-util-genai-evals

## Version 0.1.5 - 2026-01-17

### Added

- **Concurrent Evaluation Architecture** - Major refactoring for parallel evaluation processing

  - `Manager` now supports multiple worker threads with per-worker asyncio event loops
  - Configurable worker pool via `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` (default: 4)
  - Concurrent mode enabled via `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT`
  - Sequential mode remains default for backward compatibility

- **Bounded Queue Support** - Backpressure mechanism for high-throughput scenarios

  - `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` configures maximum queue size
  - Drops new items with warning when queue is full (prevents memory exhaustion)
  - Unbounded queue (default) when set to 0

- **Async Evaluation Methods** - Base `Evaluator` class now supports async evaluation

  - `evaluate_async()` - Async entry point for evaluation
  - `evaluate_llm_async()` - Async LLM invocation evaluation
  - `evaluate_agent_async()` - Async agent invocation evaluation
  - Default implementation uses `asyncio.to_thread()` for backward compatibility
  - `supports_async` property for evaluators to declare native async support

- **New Environment Variable Readers** in `env.py`

  - `read_concurrent_flag()` - Read concurrent mode flag
  - `read_worker_count()` - Read worker thread count
  - `read_queue_size()` - Read bounded queue size

- Added `gen_ai.evaluation.error` attribute to indicate errors during evaluation submission for a span.

### Changed

- `Manager._worker_count` forced to 1 in sequential mode for clarity
- Worker threads now use asyncio event loops for concurrent task processing

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-util-genai-evals
