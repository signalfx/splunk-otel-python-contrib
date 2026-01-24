# Design: Running GenAI EvaluationManager in a Separate Process

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Goals & Requirements](#goals--requirements)
4. [Current Architecture](#current-architecture)
5. [IPC/RPC Options Analysis](#ipcrpc-options-analysis)
6. [Proposed Architecture](#proposed-architecture)
7. [Detailed Design](#detailed-design)
8. [Implementation Plan](#implementation-plan)
9. [Configuration & Environment Variables](#configuration--environment-variables)
10. [Security Considerations](#security-considerations)
11. [Testing Strategy](#testing-strategy)
12. [Rollout & Migration](#rollout--migration)

---

## Overview

This document proposes a design for running the GenAI EvaluationManager in a separate process from the instrumented application. The goal is to isolate evaluation logic (including DeepEval and custom evaluators) to prevent accidental instrumentation pollution and performance interference with the customer's application.

### Related Documentation
- [README.md](../README.md) - Core concepts, emitter architecture, evaluation system
- [README.packages.architecture.md](../README.packages.architecture.md) - Package architecture, interfaces, lifecycle diagrams
- [AGENTS.md](../AGENTS.md) - Quick reference for development

## Problem Statement

### Current Issues

1. **Accidental Instrumentation Pollution**: The evaluation manager runs in the same process as the instrumented application. This causes:
   - DeepEval's internal OpenAI calls getting instrumented alongside application telemetry
   - Evaluator telemetry (spans from `deepeval.evaluate()` → OpenAI) mixing with customer application telemetry
   - Potential circular instrumentation loops when evaluators use instrumented LLM clients

2. **Performance Risks**: Running evaluations in-process creates:
   - Blocking or resource contention with application code
   - Memory pressure from evaluation models and LLM-as-judge calls
   - CPU spikes affecting application latency during batch evaluations

3. **Isolation Concerns**:
   - Customer telemetry affected by evaluation failures
   - No ability to rate-limit or throttle evaluations independently
   - Difficult to debug evaluation issues vs application issues
   - `DEEPEVAL_TELEMETRY_OPT_OUT` helps but doesn't prevent OpenAI instrumentation

### Business Impact
- Customer complaints about unexpected telemetry (evaluator spans/traces polluting dashboards)
- Potential SLA violations due to evaluation overhead in latency-sensitive applications
- Security concerns with mixed telemetry contexts

---

## Goals & Requirements

### Primary Goals
1. **Process Isolation**: Run EvaluationManager in a separate Python process with `OTEL_SDK_DISABLED=true`
2. **Bi-directional Communication**: 
   - Send GenAI invocations from instrumentation → evaluator process
   - Receive `EvaluationResult` objects from evaluator → instrumentation process
3. **Transparent Integration**: Minimal changes to existing API surface (`CompletionCallback`, `TelemetryHandler`)
4. **Backwards Compatibility**: Support for in-process mode as fallback via environment variable

### Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Evaluation manager spawns in a separate process | Must |
| R2 | `util-genai` offers GenAI invocations for evaluation to evaluation manager | Must |
| R3 | Evaluation manager runs DeepEval or custom evaluators (via entry points) | Must |
| R4 | `EvaluationResult` objects reported back to GenAI instrumentation process | Must |
| R5 | Results flow through existing `handler.evaluation_results()` → `CompositeEmitter.on_evaluation_results()` pipeline | Must |
| R6 | Graceful degradation if evaluator process fails (fallback to in-process) | Should |
| R7 | Configuration via `OTEL_INSTRUMENTATION_GENAI_*` environment variables | Should |
| R8 | Support for both sync (`evaluate_now`) and async result delivery | Should |
| R9 | Minimal serialization overhead (pickle for dataclasses) | Should |
| R10 | Clean shutdown with result flushing (honor `wait_for_all`) | Should |

---

## Current Architecture

### Current Evaluation Lifecycle (from README.packages.architecture.md)

```
Instrumentation         Emitters (Composite)                     Evaluators
--------------          ---------------------                    ----------
with handler.start_llm_invocation() as inv:  on_start(span, metrics, ...)
    model_call()                             (spans begun, metrics prealloc)
    inv.add_output_message(...)
handler.end(inv) --------> on_end(span, metrics, content_events)
        |                        |     |         |
        |                        |     |         +--> message events/logs
        |                        |     +------------> latency / tokens metrics
        |                        +------------------> span attrs + end
        v
  CompletionCallbacks (Evaluation Manager) enqueue(inv)  <-- PROBLEM AREA
        |
  async loop ------------> evaluators.evaluate(inv) -> [EvaluationResult]
        | aggregate? (env toggle)                        ↑
        v                                       DeepEval makes OpenAI calls
handler.evaluation_results(batch|single) -> on_evaluation_results(evaluation emitters)
        |                                      (pollutes telemetry)
  evaluation events/metrics (e.g. Splunk aggregated)
        v
OTel SDK exporters send spans / metrics / logs
```

### Package Structure

```text
util/
├── opentelemetry-util-genai/               # Core: TelemetryHandler, emitters, types
│   └── src/opentelemetry/util/genai/
│       ├── handler.py                      # TelemetryHandler facade
│       ├── types.py                        # GenAI, LLMInvocation, EvaluationResult, etc.
│       ├── interfaces.py                   # EmitterProtocol, CompletionCallback protocols
│       ├── callbacks.py                    # CompletionCallback registration
│       └── emitters/
│           ├── composite.py                # CompositeEmitter (chains + fan-out)
│           └── evaluation.py               # EvaluationMetricsEmitter, EvaluationEventsEmitter
│
├── opentelemetry-util-genai-evals/         # Async evaluation manager & registry
│   └── src/opentelemetry/util/genai/evals/
│       ├── manager.py                      # Manager (CompletionCallback) - TARGET FOR CHANGE
│       ├── base.py                         # Evaluator abstract base class
│       ├── registry.py                     # Evaluator entry point discovery
│       ├── bootstrap.py                    # Entry point for completion_callbacks group
│       └── env.py                          # Environment variable parsing
│
└── opentelemetry-util-genai-evals-deepeval/  # DeepEval metrics integration
    └── src/opentelemetry/util/evaluator/
        ├── deepeval.py                     # DeepevalEvaluator + registration
        ├── deepeval_runner.py              # Execution wrapper (makes OpenAI calls)
        └── deepeval_metrics.py             # Metric instantiation
```

### Key Components

#### 1. `TelemetryHandler` (`util/opentelemetry-util-genai/.../handler.py`)
- Manages GenAI invocation lifecycles (`start_llm`, `stop_llm`, `fail_llm`, etc.)
- Delegates to `CompositeEmitter` (`on_start`, `on_end`, `on_error`, `on_evaluation_results`)
- Registers completion callbacks via `register_completion_callback(cb: CompletionCallback)`
- Evaluation emission via `evaluation_results(invocation, list[EvaluationResult])`

#### 2. `Manager` (`util/opentelemetry-util-genai-evals/.../manager.py`) - Current EvaluationManager
- Implements `CompletionCallback` protocol (registered via `opentelemetry_util_genai_completion_callbacks` entry point)
- Uses internal `threading.Thread` for async evaluation (daemon worker loop)
- Maintains a `queue.Queue` for invocations (size via `OTEL_INSTRUMENTATION_GENAI_EVALUATION_QUEUE_SIZE`)
- Instantiates evaluators via `registry.get_evaluator()` (discovers `opentelemetry_util_genai_evaluators` entry points)
- Trace-id ratio sampling via `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE`

#### 3. `Evaluator` Base Class (`util/opentelemetry-util-genai-evals/.../base.py`)
- Abstract evaluator interface with `evaluate(item: GenAI) -> list[EvaluationResult]`
- Type-specific dispatch (`evaluate_llm`, `evaluate_agent`)
- Supports `default_metrics_by_type()` for per-invocation-type metric defaults

#### 4. `DeepevalEvaluator` (`util/opentelemetry-util-genai-evals-deepeval/.../deepeval.py`)
- Concrete evaluator using DeepEval library
- **Makes OpenAI calls for LLM-as-judge evaluation** ← This is the pollution source
- Returns `EvaluationResult` objects with scores, labels, explanations
- Registered via `opentelemetry_util_genai_evaluators` entry point

### Entry Points (pyproject.toml)

```toml
# opentelemetry-util-genai-evals/pyproject.toml
[project.entry-points.opentelemetry_util_genai_completion_callbacks]
evaluation_manager = "opentelemetry.util.genai.evals.bootstrap:EvaluatorCompletionCallback"

# opentelemetry-util-genai-evals-deepeval/pyproject.toml
[project.entry-points.opentelemetry_util_genai_evaluators]
deepeval = "opentelemetry.util.evaluator.deepeval:register"
```

### Data Flow Types
- **Input**: `GenAI` objects (`LLMInvocation`, `AgentInvocation`, `Workflow`, `ToolCall`, `EmbeddingInvocation`)
- **Output**: `EvaluationResult` (metric_name, score, label, explanation, error, attributes)

---

## IPC/RPC Options Analysis

### Option 1: Python `multiprocessing.Pipe` (Recommended)

**Description**: Use Python's built-in `multiprocessing.Pipe()` for duplex communication between parent and child processes.

**Pros**:
- Built into Python standard library (no extra dependencies)
- Duplex (bi-directional) by default
- Simple API (`send()` / `recv()`)
- Uses pickle for serialization (compatible with our dataclasses)
- Low latency for local IPC

**Cons**:
- Single connection (no multiple clients)
- Blocking `recv()` requires threading or polling
- Platform-specific behavior (Unix vs Windows)

**Implementation Complexity**: Low

```python
from multiprocessing import Process, Pipe

parent_conn, child_conn = Pipe(duplex=True)
p = Process(target=evaluator_worker, args=(child_conn,))
p.start()

# Parent sends invocation
parent_conn.send(invocation)

# Parent receives results
results = parent_conn.recv()
```

### Option 2: Python `multiprocessing.Queue` (Pair)

**Description**: Use two `multiprocessing.Queue` objects for request/response.

**Pros**:
- Thread and process safe
- Supports multiple producers/consumers
- Blocking and non-blocking operations
- Part of standard library

**Cons**:
- Need two queues for bi-directional communication
- Slightly more overhead than Pipe
- Result correlation requires request IDs

**Implementation Complexity**: Low-Medium

```python
from multiprocessing import Process, Queue

request_queue = Queue()
result_queue = Queue()
p = Process(target=evaluator_worker, args=(request_queue, result_queue))
```

### Option 3: `multiprocessing.managers.BaseManager`

**Description**: Use a manager server process with registered methods.

**Pros**:
- Higher-level abstraction
- Supports complex object proxies
- Can expose methods remotely
- Supports network distribution (future)

**Cons**:
- More complex setup
- Higher overhead
- Proxies can be slower than direct serialization

**Implementation Complexity**: Medium

### Option 4: gRPC

**Description**: Use Protocol Buffers and gRPC for structured RPC.

**Pros**:
- Well-defined schema (proto files)
- Bi-directional streaming support
- Language-agnostic (future polyglot support)
- Battle-tested in production systems

**Cons**:
- External dependency (`grpcio`, `grpcio-tools`)
- Requires proto compilation step
- Higher complexity for this use case
- Overkill for single-machine IPC

**Implementation Complexity**: High

### Option 5: Unix Domain Sockets + Custom Protocol

**Description**: Raw socket communication with JSON or pickle serialization.

**Pros**:
- Maximum control
- Low-level performance tuning
- Works across restarts

**Cons**:
- Requires implementing framing, serialization
- Error handling complexity
- Platform-specific (Unix sockets not available on Windows)

**Implementation Complexity**: High

### Option 6: ZeroMQ (pyzmq)

**Description**: Lightweight messaging library with various patterns.

**Pros**:
- Multiple patterns (REQ/REP, PUB/SUB, PUSH/PULL)
- High performance
- Good for complex topologies

**Cons**:
- External dependency
- Learning curve for patterns
- Overkill for simple parent-child IPC

**Implementation Complexity**: Medium

### Recommendation: Option 1 (Pipe) with Option 2 (Queue) Fallback

**Rationale**:
1. **No external dependencies** - Uses only Python standard library (aligns with project philosophy)
2. **Simple API** - Easy to implement and maintain
3. **Sufficient performance** - Pickle serialization works well with our dataclasses
4. **Duplex communication** - Pipe provides bi-directional channel
5. **Fallback path** - Queue can serve as robust alternative
6. **Cross-platform** - Works on Linux, macOS, and Windows (Python 3.10+)

---

## Proposed Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Application Process                                   │
│                                                                              │
│  ┌─────────────────┐      ┌──────────────────┐      ┌───────────────────┐  │
│  │   Instrumented  │      │  TelemetryHandler │      │ EvalManagerProxy  │  │
│  │   Application   │─────▶│                   │─────▶│ (CompletionCallback│ │
│  │   (LangChain,   │      │  start/stop/fail  │      │  registered via   │  │
│  │    OpenAI, etc) │      │  + CompositeEmitter│     │  entry point)     │  │
│  └─────────────────┘      └──────────────────┘      └─────────┬─────────┘  │
│                                    │                           │            │
│                                    │                           │ Pipe       │
│                                    ▼                           │ (duplex)   │
│  ┌───────────────────────────────────────────────────────────┐ │            │
│  │  CompositeEmitter.on_evaluation_results()                 │◀┘            │
│  │  → EvaluationMetricsEmitter (gen_ai.evaluation.*)         │              │
│  │  → EvaluationEventsEmitter (log events)                   │              │
│  │  → SplunkEvaluationAggregator (if splunk emitter enabled) │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    OpenTelemetry SDK (Spans, Metrics, Logs)          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                                                  │
                                                                  │ IPC Channel
                                                                  │ (multiprocessing.Pipe)
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Evaluator Process (OTEL_SDK_DISABLED=true)                   │
│                                                                              │
│  ┌────────────────────┐     ┌──────────────────┐     ┌──────────────────┐  │
│  │  EvalWorker        │────▶│  Manager         │────▶│ DeepevalEvaluator│  │
│  │  (IPC listener,    │     │  (existing logic,│     │ (makes OpenAI    │  │
│  │   message loop)    │     │   reused in-proc)│     │  calls - SAFE!)  │  │
│  └────────────────────┘     └──────────────────┘     └──────────────────┘  │
│           │                          │                                      │
│           │                          │ Discovers evaluators via             │
│           │                          │ opentelemetry_util_genai_evaluators  │
│           │                          │ entry point                          │
│           ▼                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  No OTel SDK active → DeepEval/OpenAI calls produce NO telemetry    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Updated Evaluation Lifecycle

```
Instrumentation         Emitters (Composite)                     Evaluator Process
--------------          ---------------------                    ------------------
with handler.start_llm_invocation() as inv:  on_start(span, metrics, ...)
    model_call()                             (spans begun, metrics prealloc)
    inv.add_output_message(...)
handler.end(inv) --------> on_end(span, metrics, content_events)
        |                        |     |         |
        |                        |     |         +--> message events/logs
        |                        |     +------------> latency / tokens metrics
        |                        +------------------> span attrs + end
        v
  CompletionCallbacks
        |
  EvalManagerProxy.on_completion(inv)
        |
        +-- serialize(inv) --> Pipe.send() ---------> EvalWorker receives
        |                                                    |
        |   [async result receiver thread]                   v
        |                                             Manager.evaluate_now(inv)
        |                                                    |
        |                                             evaluators.evaluate(inv)
        |                                                    |
        |                                             (DeepEval makes OpenAI calls)
        |                                             (NO INSTRUMENTATION!)
        |                                                    |
        +<-- Pipe.recv() <--- serialize(results) <-----------+
        |
        v
handler.evaluation_results(inv, list[EvaluationResult])
        |
        v
CompositeEmitter.on_evaluation_results(evaluation emitters)
        |
        +---> EvaluationMetricsEmitter (gen_ai.evaluation.* histograms)
        +---> EvaluationEventsEmitter (log events per result)
        v
OTel SDK exporters send spans / metrics / logs
```

### Component Responsibilities

| Component | Package | Responsibility |
|-----------|---------|----------------|
| `EvalManagerProxy` | `opentelemetry-util-genai-evals` | Implements `CompletionCallback`, serializes invocations, sends via IPC, receives results, calls `handler.evaluation_results()` |
| `EvalWorker` | `opentelemetry-util-genai-evals` | Runs in child process, listens for invocations, delegates to `Manager`, returns serialized results |
| `Manager` (existing) | `opentelemetry-util-genai-evals` | Existing evaluation logic, reused in child process for `evaluate_now()` |
| `NoOpTelemetryHandler` | `opentelemetry-util-genai-evals` | Stub handler for child process (no telemetry emission) |

---

## Detailed Design

### New Classes

#### 1. `EvalManagerProxy` (replaces `Manager` in application process)

```python
# util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/proxy.py

from __future__ import annotations

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import threading
from typing import TYPE_CHECKING, Optional
import os

from opentelemetry.util.genai.interfaces import CompletionCallback
from opentelemetry.util.genai.types import (
    GenAI,
    EvaluationResult,
    LLMInvocation,
    AgentInvocation,
    InputMessage,
    OutputMessage,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai.handler import TelemetryHandler


class EvalManagerProxy(CompletionCallback):
    """Proxy that forwards evaluations to a separate process.
    
    This class implements the CompletionCallback protocol and is registered
    via the `opentelemetry_util_genai_completion_callbacks` entry point.
    When the application process completes an LLM/Agent invocation, this
    proxy serializes the invocation data and sends it to a child process
    where the actual evaluation runs in isolation from OpenTelemetry
    instrumentation.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        interval: float | None = None,
        aggregate_results: bool | None = None,
    ) -> None:
        self._handler = handler
        self._interval = interval
        self._aggregate_results = aggregate_results
        
        # IPC channels
        self._parent_conn: Optional[Connection] = None
        self._child_conn: Optional[Connection] = None
        self._worker_process: Optional[Process] = None
        
        # Result receiver thread
        self._result_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Pending evaluations (for correlation)
        self._pending: dict[str, GenAI] = {}
        self._pending_lock = threading.Lock()
        
        self._start_worker()

    def _start_worker(self) -> None:
        """Spawn the evaluator worker process."""
        self._parent_conn, self._child_conn = Pipe(duplex=True)
        
        # Prepare configuration to pass to child
        config = {
            "interval": self._interval,
            "aggregate_results": self._aggregate_results,
            "evaluators_config": os.environ.get(
                "OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS", ""
            ),
        }
        
        self._worker_process = Process(
            target=_evaluator_worker_main,
            args=(self._child_conn, config),
            daemon=True,
            name="genai-evaluator-worker",
        )
        self._worker_process.start()
        
        # Start result receiver thread
        self._result_thread = threading.Thread(
            target=self._result_receiver_loop,
            daemon=True,
            name="genai-eval-result-receiver",
        )
        self._result_thread.start()

    def on_completion(self, invocation: GenAI) -> None:
        """Forward invocation to evaluator process.
        
        This method is called by TelemetryHandler after completing an
        LLM or Agent invocation. It serializes the invocation data
        (excluding non-picklable objects like Span and context_token)
        and sends it to the worker process via IPC.
        """
        if not self._worker_process or not self._worker_process.is_alive():
            return
            
        # Serialize invocation data (not the Span object)
        serializable = self._serialize_invocation(invocation)
        
        with self._pending_lock:
            self._pending[str(invocation.run_id)] = invocation
            
        try:
            self._parent_conn.send(("evaluate", serializable))
        except Exception:
            with self._pending_lock:
                self._pending.pop(str(invocation.run_id), None)

    def _serialize_invocation(self, invocation: GenAI) -> dict:
        """Convert invocation to serializable dict.
        
        Extracts only picklable data from the GenAI object, excluding
        Span objects, context tokens, and other non-serializable items.
        """
        return {
            "run_id": str(invocation.run_id),
            "type": type(invocation).__name__,
            "provider": invocation.provider,
            "attributes": dict(invocation.attributes) if invocation.attributes else {},
            # Add type-specific fields
            **self._extract_type_specific_fields(invocation),
        }

    def _extract_type_specific_fields(self, invocation: GenAI) -> dict:
        """Extract type-specific evaluation data."""
        fields = {}
        if isinstance(invocation, LLMInvocation):
            fields["input_messages"] = [
                {"role": m.role, "content": m.content} 
                for m in (invocation.input_messages or [])
            ]
            fields["output_messages"] = [
                {"role": m.role, "content": m.content}
                for m in (invocation.output_messages or [])
            ]
            fields["request_model"] = invocation.request_model
        elif isinstance(invocation, AgentInvocation):
            fields["operation"] = invocation.operation
            fields["name"] = invocation.name
            # Add agent-specific fields
        return fields

    def _result_receiver_loop(self) -> None:
        """Receive evaluation results from worker process."""
        while not self._shutdown_event.is_set():
            try:
                if self._parent_conn.poll(timeout=0.1):
                    message = self._parent_conn.recv()
                    self._handle_result_message(message)
            except EOFError:
                break
            except Exception:
                continue

    def _handle_result_message(self, message: tuple) -> None:
        """Process result message from worker.
        
        Correlates results with pending invocations by run_id and
        calls handler.evaluation_results() to emit telemetry via
        the CompositeEmitter's evaluation emitters.
        """
        msg_type, payload = message
        
        if msg_type == "results":
            run_id = payload["run_id"]
            results = payload["results"]
            
            with self._pending_lock:
                invocation = self._pending.pop(run_id, None)
                
            if invocation:
                # Reconstruct EvaluationResult objects
                eval_results = [
                    EvaluationResult(**r) for r in results
                ]
                # Emit telemetry via TelemetryHandler → CompositeEmitter
                self._handler.evaluation_results(invocation, eval_results)

    def shutdown(self) -> None:
        """Gracefully shutdown the evaluator process."""
        self._shutdown_event.set()
        
        if self._parent_conn:
            try:
                self._parent_conn.send(("shutdown", None))
            except Exception:
                pass
                
        if self._worker_process:
            self._worker_process.join(timeout=5.0)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
```

#### 2. `EvalWorker` (runs in evaluator process)

```python
# util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/worker.py

from __future__ import annotations

from multiprocessing.connection import Connection
import os
from typing import TYPE_CHECKING

from opentelemetry.util.genai.types import (
    GenAI,
    LLMInvocation,
    AgentInvocation,
    InputMessage,
    OutputMessage,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai.evals.manager import Manager


def _evaluator_worker_main(conn: Connection, config: dict) -> None:
    """Main entry point for evaluator worker process.
    
    This function is the target for the child process. It disables
    OpenTelemetry SDK and DeepEval telemetry to prevent any LLM calls
    made during evaluation from polluting the application's telemetry.
    """
    # CRITICAL: Disable OpenTelemetry auto-instrumentation in this process
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    
    # Import after setting env vars to ensure they take effect
    from opentelemetry.util.genai.evals.manager import Manager
    from opentelemetry.util.genai.evals._noop_handler import NoOpTelemetryHandler
    
    # Create a no-op handler - we don't emit telemetry from this process
    noop_handler = NoOpTelemetryHandler()
    
    evaluator_manager = Manager(
        noop_handler,
        interval=config.get("interval"),
        aggregate_results=config.get("aggregate_results"),
    )
    
    worker = EvalWorker(conn, evaluator_manager)
    worker.run()


class EvalWorker:
    """Worker that processes evaluation requests in the child process.
    
    Receives serialized GenAI invocations from the parent process,
    delegates to the existing Manager.evaluate_now() for evaluation,
    and sends results back via IPC.
    """
    
    def __init__(self, conn: Connection, manager: Manager) -> None:
        self._conn = conn
        self._manager = manager
        self._running = True

    def run(self) -> None:
        """Main worker loop - poll for messages and process them."""
        while self._running:
            try:
                if self._conn.poll(timeout=0.5):
                    message = self._conn.recv()
                    self._handle_message(message)
            except EOFError:
                break
            except Exception:
                continue

    def _handle_message(self, message: tuple) -> None:
        """Handle incoming message from parent process."""
        msg_type, payload = message
        
        if msg_type == "evaluate":
            self._process_evaluation(payload)
        elif msg_type == "shutdown":
            self._running = False
            self._manager.shutdown()

    def _process_evaluation(self, payload: dict) -> None:
        """Process an evaluation request.
        
        Reconstructs the GenAI invocation from serialized data,
        runs evaluation via Manager.evaluate_now(), and sends
        results back to the parent process.
        """
        # Reconstruct invocation from serialized data
        invocation = self._reconstruct_invocation(payload)
        
        # Run evaluation synchronously using existing Manager logic
        results = self._manager.evaluate_now(invocation)
        
        # Send results back as serializable dicts
        serialized_results = [
            {
                "metric_name": r.metric_name,
                "score": r.score,
                "label": r.label,
                "explanation": r.explanation,
                "attributes": r.attributes,
            }
            for r in results
        ]
        
        self._conn.send(("results", {
            "run_id": payload["run_id"],
            "results": serialized_results,
        }))

    def _reconstruct_invocation(self, payload: dict) -> GenAI:
        """Reconstruct GenAI invocation object from serialized data."""
        inv_type = payload["type"]
        
        if inv_type == "LLMInvocation":
            return LLMInvocation(
                provider=payload.get("provider"),
                request_model=payload.get("request_model", ""),
                input_messages=[
                    InputMessage(role=m["role"], content=m["content"])
                    for m in payload.get("input_messages", [])
                ],
                output_messages=[
                    OutputMessage(role=m["role"], content=m["content"])
                    for m in payload.get("output_messages", [])
                ],
                attributes=payload.get("attributes", {}),
            )
        elif inv_type == "AgentInvocation":
            return AgentInvocation(
                name=payload.get("name", ""),
                operation=payload.get("operation", "invoke_agent"),
                attributes=payload.get("attributes", {}),
            )
        # Add other types as needed
```

#### 3. Factory Function for Entry Point Registration

```python
# util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/__init__.py

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.util.genai.handler import TelemetryHandler
    from opentelemetry.util.genai.interfaces import CompletionCallback

# Environment variable to control process mode
OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS = (
    "OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS"
)


def create_evaluation_manager(
    handler: TelemetryHandler,
    *,
    interval: float | None = None,
    aggregate_results: bool | None = None,
) -> CompletionCallback:
    """Factory function to create appropriate evaluation manager.
    
    This function is called during entry point discovery via
    `opentelemetry_util_genai_completion_callbacks`. It returns
    either:
    
    - `EvalManagerProxy` (separate process mode, default when available)
    - `Manager` (in-process mode, fallback)
    
    The separate process mode is preferred because it prevents
    DeepEval's OpenAI calls from being instrumented alongside
    the application's telemetry.
    
    Configuration via environment variables:
    - OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS: "true" (default) or "false"
    - OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS: Evaluator configuration
    
    Example:
        # Entry point in pyproject.toml:
        # [project.entry-points.opentelemetry_util_genai_completion_callbacks]
        # evaluation = "opentelemetry.util.genai.evals:create_evaluation_manager"
    """
    use_separate_process = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS, "true"
    ).lower() in ("true", "1", "yes", "on")
    
    if use_separate_process:
        try:
            from opentelemetry.util.genai.evals.proxy import EvalManagerProxy
            return EvalManagerProxy(
                handler,
                interval=interval,
                aggregate_results=aggregate_results,
            )
        except Exception:
            # Fallback to in-process if multiprocessing fails
            # (e.g., on platforms without fork support)
            pass
    
    # Fallback to in-process evaluation
    from opentelemetry.util.genai.evals.manager import Manager
    return Manager(
        handler,
        interval=interval,
        aggregate_results=aggregate_results,
    )
```

#### 4. NoOp TelemetryHandler for Child Process

```python
# util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/_noop_handler.py

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.util.genai.types import GenAI, EvaluationResult


class NoOpTelemetryHandler:
    """A no-op TelemetryHandler for the evaluator child process.
    
    This stub is used in the child process where we don't want to
    emit any telemetry. The child process only runs evaluations and
    sends results back to the parent via IPC.
    """
    
    def evaluation_results(
        self, obj: GenAI, results: list[EvaluationResult]
    ) -> None:
        """No-op: results are sent back via IPC, not emitted here."""
        pass
```

### Message Protocol

| Message Type | Direction | Payload | Description |
|--------------|-----------|---------|-------------|
| `evaluate` | Parent → Child | Serialized invocation dict | Request evaluation |
| `results` | Child → Parent | run_id + list of result dicts | Evaluation results |
| `shutdown` | Parent → Child | None | Graceful shutdown request |
| `error` | Child → Parent | run_id + error dict | Evaluation error |
| `heartbeat` | Bidirectional | timestamp | Health check (optional) |

### Serialization Strategy

Since `Span` objects and OpenTelemetry contexts are not picklable, we serialize only the data needed for evaluation:

```python
# Serializable invocation data structure
{
    "run_id": "uuid-string",
    "type": "LLMInvocation" | "AgentInvocation" | "Workflow",
    "provider": "openai",
    "request_model": "gpt-4",
    "attributes": {"key": "value"},
    
    # For LLMInvocation
    "input_messages": [{"role": "user", "content": "..."}],
    "output_messages": [{"role": "assistant", "content": "..."}],
    
    # For AgentInvocation
    "operation": "invoke_agent",
    "name": "agent_name",
}
```

---

## Implementation Plan

### Phase 1: Core IPC Infrastructure (Week 1-2)

1. **Create `EvalManagerProxy` class**
   - Implement process spawning
   - Implement Pipe-based IPC
   - Implement result receiver thread

2. **Create `EvalWorker` class**
   - Implement message loop
   - Reuse existing `Manager` evaluation logic
   - Implement graceful shutdown

3. **Create serialization utilities**
   - Invocation → dict serialization
   - Dict → invocation reconstruction
   - EvaluationResult serialization

### Phase 2: Integration (Week 2-3)

4. **Factory function and configuration**
   - Environment variable support
   - Fallback mechanism
   - Process mode detection

5. **TelemetryHandler integration**
   - Update handler to use factory
   - Maintain backwards compatibility

6. **Result handling**
   - Correlation by run_id
   - Telemetry emission from parent process

### Phase 3: Robustness (Week 3-4)

7. **Error handling**
   - Process crash recovery
   - Timeout handling
   - Queue overflow management

8. **Health monitoring**
   - Process liveness checks
   - Automatic restart on failure
   - Metrics for evaluator process health

### Phase 4: Testing & Documentation (Week 4-5)

9. **Unit tests**
   - Mock IPC for isolated testing
   - Serialization round-trip tests

10. **Integration tests**
    - End-to-end with real evaluators
    - Process lifecycle tests
    - Error scenario tests

11. **Documentation**
    - Update README
    - Configuration guide
    - Troubleshooting guide

---

## Configuration & Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS` | `true` | Enable separate process mode |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` | (existing) | Evaluator configuration |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_QUEUE_SIZE` | `100` | Max pending evaluations |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_WORKER_TIMEOUT` | `30` | Worker startup timeout (seconds) |
| `OTEL_INSTRUMENTATION_GENAI_EVALS_RESULT_TIMEOUT` | `60` | Max wait for evaluation result (seconds) |

---

## Security Considerations

1. **Process Isolation**
   - Child process runs with same user permissions
   - No network exposure (local IPC only)
   - Inherited environment (filter sensitive vars if needed)

2. **Data Sensitivity**
   - Message content may contain user prompts/responses
   - Data stays in local memory/pipes (not network)
   - Consider option to redact sensitive content in transit

3. **Resource Limits**
   - Child process inherits parent's resource limits
   - Consider adding memory/CPU limits via resource module

---

## Testing Strategy

### Unit Tests
```python
def test_serialization_round_trip():
    """Test invocation serialization/deserialization."""
    invocation = LLMInvocation(...)
    serialized = proxy._serialize_invocation(invocation)
    reconstructed = worker._reconstruct_invocation(serialized)
    assert reconstructed.request_model == invocation.request_model

def test_result_correlation():
    """Test results match pending invocations."""
    # ...

def test_process_recovery():
    """Test automatic restart on worker crash."""
    # ...
```

### Integration Tests
```python
def test_end_to_end_evaluation():
    """Test full flow: invocation → IPC → evaluation → result."""
    proxy = EvalManagerProxy(handler)
    invocation = create_test_invocation()
    proxy.on_completion(invocation)
    # Wait and verify results
    
def test_graceful_shutdown():
    """Test shutdown flushes pending evaluations."""
    # ...
```

---

## Rollout & Migration

### Phase 1: Alpha (Internal Testing)
- Enable via explicit environment variable
- Collect metrics on success rate, latency
- Test with all evaluator types

### Phase 2: Beta (Opt-in)
- Document feature in release notes
- Default remains in-process
- Gather customer feedback

### Phase 3: GA (Default Enabled)
- Flip default to separate process
- Provide escape hatch to in-process mode
- Monitor for issues in production

### Backwards Compatibility

- Existing `Manager` class remains unchanged
- Factory function abstracts implementation choice
- Environment variable controls mode selection
- All existing environment variables continue to work

---

## Appendix A: Alternative Considered - Subprocess with stdin/stdout

Using `subprocess.Popen` with JSON over stdin/stdout was considered but rejected because:
1. More complex process management
2. Manual JSON serialization needed
3. No built-in message framing
4. Less integration with multiprocessing module

---

## Appendix B: Performance Estimates

| Metric | In-Process | Separate Process |
|--------|------------|------------------|
| IPC Latency | N/A | ~0.1-1ms per message |
| Serialization | N/A | ~0.1-0.5ms for typical invocation |
| Memory Isolation | Shared | Fully isolated |
| CPU Isolation | Shared | Fully isolated |
| Failure Blast Radius | Affects app | Isolated to evaluator |

---

## References

- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Current Manager implementation](../util/opentelemetry-util-genai-evals/src/opentelemetry/util/genai/evals/manager.py)
- [DeepEval Evaluator](../util/opentelemetry-util-genai-evals-deepeval/src/opentelemetry/util/evaluator/deepeval.py)
- [TelemetryHandler](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/handler.py)
- [GenAI Types](../util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py)
- [Project Architecture](../README.packages.architecture.md)
- [Project README](../README.md)
- [Agent Instructions](../AGENTS.md)
