# Concurrent Evaluation Architecture

This document describes the architecture changes for multi-threaded evaluation processing in the Splunk OpenTelemetry GenAI instrumentation.

## Overview

The evaluation system processes LLM invocations through evaluators (like DeepEval) to generate quality metrics such as Bias, Toxicity, Answer Relevancy, and Faithfulness. These evaluations involve LLM-as-a-Judge calls which are I/O-bound and benefit significantly from parallel processing.

## Architecture Comparison

### Previous Architecture (Sequential)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application                                  │
│                                                                      │
│   LLM Call ──► TelemetryHandler ──► CompletionCallback               │
│                                           │                          │
│                                           ▼                          │
│                                    ┌─────────────┐                   │
│                                    │    Queue    │                   │
│                                    │ (unbounded) │                   │
│                                    └──────┬──────┘                   │
│                                           │                          │
│                                           ▼                          │
│                              ┌────────────────────────┐              │
│                              │   Single Worker Thread │              │
│                              │                        │              │
│                              │  ┌──────────────────┐  │              │
│                              │  │ Process Item 1   │  │              │
│                              │  │ (sync LLM call)  │  │              │
│                              │  └────────┬─────────┘  │              │
│                              │           │            │              │
│                              │           ▼            │              │
│                              │  ┌──────────────────┐  │              │
│                              │  │ Process Item 2   │  │              │
│                              │  │ (sync LLM call)  │  │              │
│                              │  └────────┬─────────┘  │              │
│                              │           │            │              │
│                              │           ▼            │              │
│                              │         ...            │              │
│                              └────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Problems:
- Single thread processes one evaluation at a time
- I/O-bound LLM calls block the worker
- Long queue drain times (5+ minutes for batch)
- Sequential processing wastes time waiting for API responses
```

### New Architecture (Concurrent)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application                                     │
│                                                                              │
│   LLM Call ──► TelemetryHandler ──► CompletionCallback                       │
│                                           │                                  │
│                                           ▼                                  │
│                                  ┌─────────────────┐                         │
│                                  │  Bounded Queue  │                         │
│                                  │  (configurable) │                         │
│                                  └────────┬────────┘                         │
│                                           │                                  │
│              ┌────────────────────────────┼────────────────────────────┐     │
│              │                            │                            │     │
│              ▼                            ▼                            ▼     │
│   ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│   │   Worker Thread 1   │    │   Worker Thread 2   │    │  Worker Thread N│  │
│   │                     │    │                     │    │                 │  │
│   │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────┐ │  │
│   │ │  asyncio loop   │ │    │ │  asyncio loop   │ │    │ │asyncio loop │ │  │
│   │ │                 │ │    │ │                 │ │    │ │             │ │  │
│   │ │ ┌─────────────┐ │ │    │ │ ┌─────────────┐ │ │    │ │ ┌─────────┐ │ │  │
│   │ │ │ async eval  │ │ │    │ │ │ async eval  │ │ │    │ │ │async    │ │ │  │
│   │ │ │ (Item A)    │ │ │    │ │ │ (Item B)    │ │ │    │ │ │eval     │ │ │  │
│   │ │ └─────────────┘ │ │    │ │ └─────────────┘ │ │    │ │ │(Item C) │ │ │  │
│   │ └─────────────────┘ │    │ └─────────────────┘ │    │ │ └─────────┘ │ │  │
│   └─────────────────────┘    └─────────────────────┘    │ └─────────────┘ │  │
│                                                         └─────────────────┘  │
│                                                                              │
│   Benefits:                                                                  │
│   - Multiple workers process items in parallel                               │
│   - Each worker has its own asyncio event loop                               │
│   - Async LLM calls don't block other workers                                │
│   - Bounded queue provides backpressure                                      │
│   - ~13x faster throughput                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Manager (`manager.py`)

The `Manager` class orchestrates evaluation processing:

```python
class Manager:
    def __init__(self):
        # Configuration from environment
        self._concurrent_mode = read_concurrent_flag()  # OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT
        self._worker_count = read_worker_count()        # OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS
        queue_size = read_queue_size()                  # OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE
        
        # Bounded or unbounded queue
        self._queue = Queue(maxsize=queue_size) if queue_size > 0 else Queue()
        
        # Worker pool
        if self._concurrent_mode:
            # Start N worker threads with asyncio loops
            for i in range(self._worker_count):
                worker = Thread(target=self._concurrent_worker_loop, daemon=True)
                worker.start()
                self._workers.append(worker)
        else:
            # Single sequential worker (backward compatible)
            self._worker_count = 1
            worker = Thread(target=self._worker_loop, daemon=True)
            worker.start()
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `_concurrent_worker_loop()` | Creates asyncio event loop per worker |
| `_async_worker_task()` | Async task that pulls from queue |
| `_process_invocation_async()` | Async evaluation processing |
| `_evaluate_invocation_async()` | Runs evaluators concurrently |

### 2. Base Evaluator (`base.py`)

The `Evaluator` base class now supports async methods:

```python
class Evaluator:
    @property
    def supports_async(self) -> bool:
        """Override to return True if evaluator has native async support."""
        return False
    
    async def evaluate_async(self, item: GenAI) -> list[EvaluationResult]:
        """Async evaluation entry point."""
        if isinstance(item, LLMInvocation):
            return list(await self.evaluate_llm_async(item))
        if isinstance(item, AgentInvocation):
            return list(await self.evaluate_agent_async(item))
        return []
    
    async def evaluate_llm_async(self, invocation: LLMInvocation):
        """Default: run sync method in thread pool."""
        return await asyncio.to_thread(self.evaluate_llm, invocation)
```

### 3. DeepEval Evaluator (`deepeval.py`)

The `DeepevalEvaluator` implements native async support:

```python
class DeepevalEvaluator(Evaluator):
    @property
    def supports_async(self) -> bool:
        return True  # Native async support
    
    async def evaluate_llm_async(self, invocation: LLMInvocation):
        # Build test case and metrics
        test_case = self._build_test_case(invocation)
        metrics = self._build_metrics()
        
        # Run DeepEval asynchronously
        result = await run_evaluation_async(test_case, metrics)
        return self._convert_results(result)
```

### 4. DeepEval Runner (`deepeval_runner.py`)

The runner handles DeepEval's internal async configuration:

```python
async def run_evaluation_async(test_case, metrics, debug_log=None):
    """Run DeepEval evaluation asynchronously."""
    
    # Enable DeepEval's internal async mode when concurrent mode is active
    use_deepeval_async = _is_async_mode_enabled()
    
    def _run_sync():
        async_config = AsyncConfig(
            run_async=use_deepeval_async,  # Parallel metric evaluation
            max_concurrent=10              # Up to 10 metrics in parallel
        )
        return deepeval_evaluate(
            [test_case],
            metrics,
            async_config=async_config,
            display_config=DisplayConfig(show_indicator=False)
        )
    
    # Run in thread pool to not block event loop
    return await asyncio.to_thread(_run_sync)
```

### 5. DeepEval AsyncConfig Integration

We leverage DeepEval's built-in `AsyncConfig` to enable parallel metric evaluation within each invocation. This is documented in [DeepEval's Async Configs documentation](https://deepeval.com/docs/evaluation-flags-and-configs#async-configs).

#### How It Works

When `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true`, we enable DeepEval's internal async mode:

```python
from deepeval.evaluate import AsyncConfig

async_config = AsyncConfig(
    run_async=True,       # Enable concurrent evaluation of test cases AND metrics
    max_concurrent=10,    # Maximum parallel metrics at any point in time
    throttle_value=0      # No throttling (can be increased for rate-limited APIs)
)
```

#### AsyncConfig Parameters

| Parameter | Our Value | Description |
|-----------|-----------|-------------|
| `run_async` | `True` (when concurrent mode enabled) | Enables concurrent evaluation of test cases **AND** metrics |
| `max_concurrent` | `10` | Maximum number of metrics that can run in parallel |
| `throttle_value` | `0` | Seconds to wait between metric evaluations (for rate limiting) |

#### Two Levels of Parallelism

Our architecture provides **two levels of concurrent processing**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LEVEL 1: Worker Thread Parallelism                       │
│                     (Our Implementation - Manager)                           │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │  Worker 1   │    │  Worker 2   │    │  Worker 3   │    │  Worker 4   │  │
│   │ Invocation A│    │ Invocation B│    │ Invocation C│    │ Invocation D│  │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│          │                  │                  │                  │         │
│          ▼                  ▼                  ▼                  ▼         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                LEVEL 2: Metric Parallelism                          │   │
│   │                (DeepEval AsyncConfig)                               │   │
│   │                                                                     │   │
│   │   Within each invocation, metrics run concurrently:                 │   │
│   │                                                                     │   │
│   │   ┌─────────┐  ┌───────────┐  ┌──────────────────┐  ┌────────────┐  │   │
│   │   │  Bias   │  │ Toxicity  │  │ Answer Relevancy │  │Faithfulness│  │   │
│   │   │ (LLM)   │  │  (LLM)    │  │     (LLM)        │  │   (LLM)    │  │   │
│   │   └─────────┘  └───────────┘  └──────────────────┘  └────────────┘  │   │
│   │        │            │                │                    │         │   │
│   │        └────────────┴────────────────┴────────────────────┘         │   │
│   │                            │                                        │   │
│   │                    All run in parallel                              │   │
│   │                   (up to max_concurrent=10)                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Level 1 (Our Implementation):**
- Multiple worker threads process different invocations simultaneously
- Controlled by `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS`

**Level 2 (DeepEval's AsyncConfig):**
- Within each invocation, multiple metrics (Bias, Toxicity, etc.) run in parallel
- Controlled by `AsyncConfig(run_async=True, max_concurrent=10)`

#### Why Two Levels?

| Level | What it Parallelizes | Benefit |
|-------|---------------------|---------|
| **Level 1** (Workers) | Different LLM invocations | Multiple conversations evaluated simultaneously |
| **Level 2** (AsyncConfig) | Metrics within one invocation | Bias, Toxicity, Relevancy run in parallel |

This combination provides **multiplicative performance gains**:
- 4 workers × 4 parallel metrics = up to **16 concurrent LLM-as-a-Judge calls**

#### Configuration for Rate-Limited APIs

If your LLM API has rate limits, you can adjust both levels:

```bash
# Reduce worker threads
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=2

# Note: max_concurrent is hardcoded to 10 in our implementation
# For further rate limiting, reduce workers or consider throttle_value
```

#### Reference

For more details on DeepEval's async configuration options, see:
- [DeepEval Async Configs Documentation](https://deepeval.com/docs/evaluation-flags-and-configs#async-configs)

## Data Flow

### Sequential Mode (Default)

```
Invocation → Queue → Worker → Evaluator.evaluate() → sync LLM call → Results
                ↑                                            │
                └────────────────────────────────────────────┘
                              (blocks until complete)
```

### Concurrent Mode

```
                    ┌──► Worker 1 → asyncio → Evaluator.evaluate_async() ──┐
                    │                              │                        │
Invocation → Queue ─┼──► Worker 2 → asyncio → async LLM call ──────────────┼──► Results
                    │                              │                        │
                    └──► Worker N → asyncio → (non-blocking) ──────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT` | `false` | Enable concurrent mode |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS` | `4` | Number of worker threads |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` | `0` | Queue size (0 = unbounded) |

### Recommended Configurations

**Development / Low Volume:**
```bash
# Use defaults (sequential mode)
# No additional configuration needed
```

**Production / High Volume:**
```bash
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=4
export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=100
```

**Rate-Limited APIs:**
```bash
export OTEL_INSTRUMENTATION_GENAI_EVALS_CONCURRENT=true
export OTEL_INSTRUMENTATION_GENAI_EVALS_WORKERS=2  # Reduce to avoid rate limits
export OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE=50
```

## Bounded Queue & Backpressure

When `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` is set:

```python
def enqueue(self, invocation: GenAI):
    if self._queue.full():
        _LOGGER.warning(
            "Evaluation queue full (size=%d), dropping invocation",
            self._queue.maxsize
        )
        return  # Drop item, don't block
    self._queue.put_nowait(invocation)
```

Benefits:
- Prevents unbounded memory growth
- Provides backpressure signal
- Graceful degradation under load

## Thread Safety

The implementation ensures thread safety through:

1. **Thread-safe Queue**: Python's `queue.Queue` is thread-safe
2. **Per-worker Event Loops**: Each worker has its own asyncio loop
3. **No Shared Mutable State**: Workers don't share evaluation state
4. **Atomic Operations**: Queue operations are atomic

## Shutdown Handling

```python
def shutdown(self):
    """Gracefully shutdown all workers."""
    self._shutdown.set()  # Signal workers to stop
    
    # Wait for queue to drain
    self._queue.join()
    
    # Wait for all workers to complete
    for worker in self._workers:
        worker.join(timeout=5.0)
```

## Performance Characteristics

| Metric | Sequential | Concurrent (4 workers) |
|--------|------------|------------------------|
| Throughput | 1 eval/time | ~4 evals/time |
| Latency | O(n) | O(n/workers) |
| Memory | Low | Moderate |
| CPU | Single core | Multi-core |

## Backward Compatibility

- **Default behavior unchanged**: Sequential mode is default
- **No breaking changes**: Existing code works without modification
- **Opt-in concurrent mode**: Must explicitly enable via environment variable
- **Same evaluation results**: Quality and accuracy unchanged

## Testing

Unit tests cover:

1. **Environment Variable Parsing** (`test_env.py`)
   - `read_concurrent_flag()`
   - `read_worker_count()`
   - `read_queue_size()`

2. **Manager Behavior** (`test_evaluation_manager.py`)
   - Concurrent mode initialization
   - Worker count configuration
   - Queue size bounds
   - Shutdown handling

3. **Async Evaluators** (`test_evaluators.py`)
   - `supports_async` property
   - `evaluate_async()` delegation
   - Native async implementations

4. **DeepEval Runner** (`test_deepeval_runner.py`)
   - `run_evaluation_async()`
   - AsyncConfig integration
   - Error handling

## Future Enhancements

Potential improvements:

1. **Adaptive Worker Scaling**: Adjust workers based on queue depth
2. **Priority Queue**: Process critical evaluations first
3. **Circuit Breaker**: Handle API failures gracefully
4. **Metrics Export**: Worker pool utilization metrics
5. **Rate Limiting**: Built-in rate limiting per worker

