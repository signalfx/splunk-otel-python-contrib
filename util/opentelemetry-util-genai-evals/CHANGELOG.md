# Changelog

All notable changes to this repository are documented in this file.

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

### Changed
- `Manager._worker_count` forced to 1 in sequential mode for clarity
- Worker threads now use asyncio event loops for concurrent task processing

## Version 0.1.4 - 2025-11-07

- Initial 0.1.4 release of splunk-otel-util-genai-evals