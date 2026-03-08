# Dynamic Configuration via OpAMP

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Upstream OpAMP Client (PR #3635)](#upstream-opamp-client-pr-3635)
5. [Reactive ConfigStore](#reactive-configstore)
6. [OpAMP Adapter Layer](#opamp-adapter-layer)
7. [Dynamically Configurable Settings](#dynamically-configurable-settings)
8. [Integration with TelemetryHandler](#integration-with-telemetryhandler)
9. [Integration with Evaluation System](#integration-with-evaluation-system)
10. [Implementation Phases](#implementation-phases)
11. [End-to-End Flows](#end-to-end-flows)
12. [Risk and Mitigation](#risk-and-mitigation)

### Related Documentation
- [README.md](../README.md) - Core concepts, emitter architecture, evaluation system
- [README.packages.architecture.md](../README.packages.architecture.md) - Package architecture, interfaces, lifecycle diagrams
- [AGENTS.md](../AGENTS.md) - Quick reference for development

---

## Overview

This document describes the architecture for adding dynamic configuration capabilities to the Splunk OTel Python GenAI agent using the [OpAMP (Open Agent Management Protocol)](https://opentelemetry.io/docs/specs/opamp/). The goal is to enable runtime changes to sampling rate, content capture, Judge model, eval metrics, and process isolation -- without process restarts.

The design uses the upstream [`opentelemetry-opamp-client`](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3635) package as the transport layer and builds a thin adapter on top that bridges OpAMP remote config into a reactive internal `ConfigStore`.

---

## Problem Statement

### Current Limitations

Configuration in the Splunk OTel Python GenAI agent is **almost entirely static** -- read once at initialization and never refreshed.

| Component | File | When Config Is Read | Hot-Reload? |
| :--- | :--- | :--- | :--- |
| Settings (emitters, sampling, capture) | `config.py` via `parse_env()` | `TelemetryHandler.__init__` | No |
| Emitter pipeline | `emitters/configuration.py` | `TelemetryHandler.__init__` | No |
| Sampling rate | `config.py` / `handler.py` | `TelemetryHandler.__init__` | No |
| Content capture mode | `utils.py` / `handler.py` | Each `start_*` call | **Yes** (partial) |
| Evaluator config | `evals/manager.py` | Evaluation manager init | No |
| Judge model | `deepeval_model.py` | Per-evaluation call | **Yes** (reads env each time) |

Changing any static setting requires a **full process restart** of the customer's AI application, which introduces downtime in production.

### What This Enables

- **Dynamic Sampling**: Drop to 1% at steady-state, auto-burst to 100% when anomalies are detected
- **Judge LLM Hot-Swap**: Switch from GPT-4o to GPT-4o-mini or Splunk Hosted Judge without restart
- **On-the-fly Content Capture**: Toggle PII/content capture during incident investigations
- **Live Eval Metrics**: Activate new eval metrics (bias, toxicity) without redeployment
- **Provider Circuit Breaking**: Failover from a failing Judge provider in milliseconds

---

## Architecture Overview

The design introduces three new components that sit between the upstream OpAMP client and the existing GenAI instrumentation system:

```
                                ┌─────────────────────────────────┐
                                │        OpAMP Server             │
                                │   (Splunk Observability Cloud)  │
                                └────────────┬────────────────────┘
                                             │ protobuf / HTTP
                                             │
                     ┌───────────────────────────────────────────────┐
                     │       Upstream: opentelemetry-opamp-client    │
                     │                                               │
                     │  ┌──────────────┐   ┌───────────────────┐    │
                     │  │  OpAMPAgent   │   │   OpAMPClient     │    │
                     │  │  (threads,    │──▶│   (protobuf       │    │
                     │  │   heartbeat,  │   │    messages,      │    │
                     │  │   retry)      │   │    send/receive)  │    │
                     │  └──────┬───────┘   └───────────────────┘    │
                     │         │ calls message_handler               │
                     └─────────┼─────────────────────────────────────┘
                               │
                     ┌─────────▼─────────────────────────────────────┐
                     │       New: Adapter Layer                      │
                     │       (opentelemetry-util-genai-opamp)        │
                     │                                               │
                     │  ┌──────────────────┐  ┌─────────────────┐   │
                     │  │ splunk_opamp_     │  │ ConfigKeyMapper │   │
                     │  │ handler()        │─▶│ (OpAMP keys →   │   │
                     │  │ (message_handler │  │  ConfigStore     │   │
                     │  │  callback)       │  │  keys)           │   │
                     │  └──────────────────┘  └────────┬────────┘   │
                     │                                  │            │
                     │  ┌──────────────────┐            │            │
                     │  │ OpAMPBootstrap   │            │            │
                     │  │ (lifecycle mgmt) │            │            │
                     │  └──────────────────┘            │            │
                     └──────────────────────────────────┼────────────┘
                                                        │ update()
                     ┌──────────────────────────────────▼────────────┐
                     │       New: Reactive Config Layer               │
                     │       (in opentelemetry-util-genai)            │
                     │                                               │
                     │  ┌────────────────────────────────────────┐   │
  env vars ─────────▶│  │            ConfigStore                 │   │
  (initial load)     │  │  - thread-safe (RLock)                 │   │
                     │  │  - observable (subscribe/notify)        │   │
                     │  │  - atomic batch updates                 │   │
                     │  └──────┬──────┬──────┬──────┬────────────┘   │
                     │         │      │      │      │                │
                     └─────────┼──────┼──────┼──────┼────────────────┘
                      observers│      │      │      │
                     ┌─────────▼──┐ ┌─▼────┐ │  ┌───▼──────────┐
                     │ Sampling   │ │Capture│ │  │ Eval Manager │
                     │ Rate       │ │ Mode  │ │  │ (reconfig)   │
                     └────────────┘ └──────┘ │  └──────────────┘
                                          ┌──▼─────────┐
                                          │ Judge Model│
                                          │ (hot-swap) │
                                          └────────────┘
```

### Design Principles

| Principle | How It's Applied |
| :--- | :--- |
| **Backward compatible** | Env vars still work. ConfigStore reads them at init. OpAMP is opt-in. |
| **Thread-safe** | `RLock`-guarded store; callbacks invoked outside lock; atomic batch updates. |
| **No restart needed** | Observers react to changes in-place. In-flight evaluations complete before switching. |
| **Minimal coupling** | ConfigStore is standalone. OpAMP adapter is a separate package. Core util has no OpAMP dependency. |
| **Observer pattern** | Decouples config source from consumers. Future config sources (UI, API, file watch) require no consumer changes. |

---

## Upstream OpAMP Client (PR #3635)

The upstream [`opentelemetry-opamp-client`](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3635) provides the transport layer. It is consumed as an **external dependency** -- not vendored.

### What It Provides

| Component | Purpose |
| :--- | :--- |
| `OpAMPClient` | Builds and sends protobuf messages (connection, heartbeat, disconnect, config status, effective config, full state) |
| `OpAMPAgent` | Manages worker/scheduler daemon threads, retry with exponential backoff + jitter, heartbeat at configurable interval |
| `RequestsTransport` | HTTP transport using the `requests` library, supports mTLS |
| `messages` module | Protobuf serialization helpers for AgentToServer / ServerToAgent |

### Capabilities Implemented by Upstream

- `ReportsStatus` -- reports agent description on connection
- `ReportsHeartbeat` -- periodic heartbeat
- `AcceptsRemoteConfig` -- receives remote config from server
- `ReportsRemoteConfig` -- acknowledges config status back
- `ReportsEffectiveConfig` -- reports current effective config

### Integration Model

The upstream client uses a **callback pattern**. Distros provide a `message_handler` function:

```python
def message_handler(
    agent: OpAMPAgent,
    client: OpAMPClient,
    message: opamp_pb2.ServerToAgent
) -> None:
    # Distro-specific config application logic
    ...
```

This is the extension point where the adapter layer connects.

### Dependencies

| Dependency | Version | Conflict with Splunk Distro? |
| :--- | :--- | :--- |
| `opentelemetry-api` | `~= 1.12` | No (Splunk uses `>=1.31.0`, which is a subset) |
| `protobuf` | `>=5.0, <7.0` | No (new dependency, no conflict) |
| `uuid-utils` | `>=0.11.0, <1` | No (new dependency) |
| `requests` | (transport) | No (commonly available) |

---

## Reactive ConfigStore

The `ConfigStore` is the central reactive configuration layer. It lives in the core `opentelemetry-util-genai` package and has **no OpAMP dependency**.

### Location

`util/opentelemetry-util-genai/src/opentelemetry/util/genai/config_store.py`

### API Surface

```python
class ConfigStore:
    def get(self, key: str) -> Any:
        """Thread-safe read of a config value."""

    def set(self, key: str, value: Any) -> None:
        """Thread-safe write of a single config value. Notifies observers if changed."""

    def update(self, updates: dict[str, Any]) -> None:
        """Atomic batch update. Observers see consistent state."""

    def subscribe(self, key: str, callback: Callable[[str, Any, Any], None]) -> str:
        """Register observer. Returns subscription ID. Callback: (key, old, new)."""

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove observer."""

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe snapshot of all current values."""
```

### Thread Safety Model

- Internal `dict` guarded by `threading.RLock`
- Reads and writes acquire the lock
- `update()` applies all changes atomically under a single lock acquisition
- Observer callbacks are collected under the lock but **invoked outside the lock** to prevent deadlocks
- No-change optimization: observers are not notified if the new value equals the old value

### Config Keys

| Key | Type | Source Env Var |
| :--- | :--- | :--- |
| `evaluation_sample_rate` | `float` (0.0-1.0) | `OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE` |
| `capture_message_content` | `bool` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` |
| `capture_message_content_mode` | `str` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE` |
| `eval_evaluators` | `str` | `OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS` |
| `eval_separate_process` | `bool` | `OTEL_INSTRUMENTATION_GENAI_EVALS_SEPARATE_PROCESS` |
| `judge_model` | `str` | `DEEPEVAL_JUDGE_MODEL` |
| `judge_base_url` | `str` | `DEEPEVAL_LLM_BASE_URL` |
| `judge_provider` | `str` | `DEEPEVAL_LLM_PROVIDER` |
| `eval_metrics_list` | `str` | `SPLUNK_OTEL_GENAI_EVALS_METRICS_LIST` |

### Initialization

At startup, `populate_store(store)` in `config.py` reads environment variables and populates the store. The existing `Settings` dataclass and `parse_env()` remain unchanged for backward compatibility.

---

## OpAMP Adapter Layer

A new package that bridges the upstream OpAMP client to the internal ConfigStore.

### Package Structure

```
util/opentelemetry-util-genai-opamp/
├── pyproject.toml
├── src/opentelemetry/util/genai/opamp/
│   ├── __init__.py
│   ├── bootstrap.py         # OpAMPBootstrap lifecycle class
│   ├── handler.py            # splunk_opamp_handler callback
│   ├── key_mapper.py         # OpAMP config key → ConfigStore key mapping
│   └── version.py
└── tests/
    ├── test_handler.py
    ├── test_key_mapper.py
    └── test_bootstrap.py
```

### Dependencies

```toml
dependencies = [
    "splunk-otel-util-genai>=0.2.0",
    "opentelemetry-opamp-client>=0.1.0",
]
```

### Key Components

**`key_mapper.py`** -- Translates OpAMP remote config JSON keys to ConfigStore keys, with type coercion and validation:

```
OpAMP JSON                    ConfigStore Key                Type
─────────────────────────────────────────────────────────────────
sampling_rate              →  evaluation_sample_rate         float (clamped 0.0-1.0)
capture_message_content    →  capture_message_content        bool
capture_message_content_mode → capture_message_content_mode  str (validated enum)
judge_model                →  judge_model                    str
judge_base_url             →  judge_base_url                 str
judge_provider             →  judge_provider                 str
eval_metrics_list          →  eval_metrics_list              str
eval_evaluators            →  eval_evaluators                str
eval_separate_process      →  eval_separate_process          bool
```

**`handler.py`** -- The `message_handler` callback passed to `OpAMPAgent`. It:
1. Checks `ServerToAgent.flags` for `ReportFullState` and responds accordingly
2. Decodes remote config using `OpAMPClient.decode_remote_config()`
3. Maps keys via `ConfigKeyMapper`
4. Pushes updates to `ConfigStore` via `update()`
5. Acknowledges config status back to server
6. Reports effective config from current ConfigStore snapshot

The upstream `message_handler` signature is `(agent, client, message)`. The `config_store` reference is bound using `functools.partial`.

**`bootstrap.py`** -- `OpAMPBootstrap` manages lifecycle:
- `start(endpoint, service_name, **kwargs)` -- creates `OpAMPClient` and `OpAMPAgent`, starts the agent
- `stop()` -- graceful shutdown (sends `AgentDisconnect`)
- Activated in `TelemetryHandler.__init__` when `OTEL_INSTRUMENTATION_GENAI_OPAMP_ENDPOINT` is set **and** the package is installed

---

## Dynamically Configurable Settings

### What Changes at Runtime

| Setting | Observer Location | What Happens on Change |
| :--- | :--- | :--- |
| **Sampling Rate** | `TelemetryHandler` | `self._sampler = TraceIdRatioBased(new_rate)` -- next `_should_sample_for_evaluation()` uses new rate |
| **Content Capture** | `TelemetryHandler` | `_refresh_capture_content()` reads from ConfigStore instead of env vars |
| **Content Capture Mode** | `TelemetryHandler` | Same as above -- emitter flags update on next `start_*` call |
| **Eval Evaluators** | `Manager` | `reconfigure()` reloads plans and reinstantiates evaluators (in-flight evals complete first) |
| **Eval Separate Process** | `Manager` | `reconfigure()` stops current workers, restarts in new mode |
| **Judge Model** | `DeepevalEvaluator` | Next evaluation reads from ConfigStore (already per-call, just changes source) |
| **Judge Base URL** | `DeepevalEvaluator` | Same as above |
| **Eval Metrics List** | `Manager` | `reconfigure()` updates active metrics |

### What Stays Static

The emitter pipeline (`CompositeEmitter`, `SpanEmitter`, `MetricsEmitter`, etc.) is NOT dynamically reconfigurable. Changing emitters requires a restart. This is intentional -- emitter changes involve tracer/meter/logger provider wiring that cannot safely be swapped mid-flight.

---

## Integration with TelemetryHandler

### Changes to `handler.py`

1. **ConfigStore creation** in `__init__`:
   - Create `self._config_store = ConfigStore()`
   - Call `populate_store(self._config_store)` after `parse_env()`
   - Register observers for sampling rate, content capture, content capture mode

2. **Thread-safe singleton** in `get_telemetry_handler()`:
   - Current implementation uses `getattr`/`setattr` on the function object with no lock
   - Add `threading.Lock` to prevent double-creation under concurrent access

3. **Sampling rate observer**:
   - On `evaluation_sample_rate` change, replace `self._sampler` with a new `TraceIdRatioBased` instance
   - `_should_sample_for_evaluation()` already reads `self._sampler` each time -- no further changes needed

4. **Content capture observer**:
   - Refactor `_refresh_capture_content()` to read from `self._config_store` instead of direct `os.environ.get()` calls
   - Keep calling it on each `start_*` as today -- ConfigStore is just the new source of truth

5. **OpAMP opt-in bootstrap** at end of `__init__`:
   - Check for `OTEL_INSTRUMENTATION_GENAI_OPAMP_ENDPOINT`
   - Lazy-import `OpAMPBootstrap` (handles missing package gracefully)
   - Start the OpAMP agent

---

## Integration with Evaluation System

### Changes to `manager.py`

1. **`reconfigure(updates: dict)` method**:
   - Accepts a dict of changed config keys
   - For `eval_evaluators`: reload plans via `_load_plans()` and reinstantiate evaluators
   - For `eval_separate_process`: stop current workers, restart in new mode
   - Graceful transition: in-flight evaluations complete before switching

2. **ConfigStore subscription**:
   - Manager subscribes to relevant keys via handler's config_store
   - Observer calls `reconfigure()` with the changed key/value

### Changes to DeepEval Model Resolution

- `_default_model()` already reads env per evaluation call, so it is partially dynamic
- Add ConfigStore-aware path: check ConfigStore first, fall back to env vars
- Enables hot-swap without env var mutation (thread-safe, no race conditions)

---

## Implementation Phases

### Phase 1: Reactive ConfigStore (Foundation)

No external dependencies. Pure internal refactoring.

- Create `config_store.py` with `ConfigStore` class
- Define config key constants
- Add `populate_store()` to `config.py`
- Unit tests: thread safety, observer notifications, atomic updates, no-change skipping

### Phase 2: Wire ConfigStore into TelemetryHandler

- Integrate ConfigStore into `TelemetryHandler.__init__`
- Make `get_telemetry_handler()` thread-safe
- Sampling rate observer
- Content capture observer (refactor `_refresh_capture_content()`)
- Tests for config change propagation

### Phase 3: Wire ConfigStore into Evaluation System

- `Manager.reconfigure()` method
- ConfigStore-aware judge model resolution
- Tests for evaluator reconfiguration and judge model hot-swap

### Phase 4: OpAMP Adapter Package

Depends on Phase 2 + Phase 3 and the upstream PR merging.

- Create `opentelemetry-util-genai-opamp` package
- Implement `key_mapper.py`, `handler.py`, `bootstrap.py`
- Wire into `TelemetryHandler` (opt-in via env var + package install)
- Tests: mock OpAMP server, config flow, lifecycle

### Phase 5: Documentation and Polish

- Update README.md, README.packages.architecture.md, AGENTS.md
- Add `OTEL_INSTRUMENTATION_GENAI_OPAMP_ENDPOINT` to environment_variables.py
- Update CHANGELOG.md for all modified packages
- Version bumps

### Dependency Graph

```
Phase 1 ──▶ Phase 2 ──┐
              │        ├──▶ Phase 4 ──▶ Phase 5
Phase 1 ──▶ Phase 3 ──┘
```

Phase 2 and 3 can execute in parallel after Phase 1.

---

## End-to-End Flows

### Flow 1: Sampling Rate Change via OpAMP

```
OpAMP Server                  OpAMP Client           Adapter              ConfigStore          TelemetryHandler
    │                             │                     │                      │                      │
    │  ServerToAgent              │                     │                      │                      │
    │  (remote_config:            │                     │                      │                      │
    │   sampling_rate=0.01)       │                     │                      │                      │
    │────────────────────────────▶│                     │                      │                      │
    │                             │  message_handler()  │                      │                      │
    │                             │────────────────────▶│                      │                      │
    │                             │                     │  update(             │                      │
    │                             │                     │   {eval_sample_rate: │                      │
    │                             │                     │    0.01})            │                      │
    │                             │                     │────────────────────▶│                      │
    │                             │                     │                      │  on_change(           │
    │                             │                     │                      │   eval_sample_rate,   │
    │                             │                     │                      │   1.0 → 0.01)        │
    │                             │                     │                      │─────────────────────▶│
    │                             │                     │                      │                      │
    │                             │                     │                      │     self._sampler =   │
    │                             │                     │                      │     TraceIdRatioBased │
    │                             │                     │                      │     (0.01)            │
    │                             │                     │                      │                      │
    │  AgentToServer              │                     │                      │                      │
    │  (config_status: APPLIED)   │                     │                      │                      │
    │◀────────────────────────────│                     │                      │                      │
```

### Flow 2: Judge Model Hot-Swap

```
OpAMP Server                  Adapter              ConfigStore          DeepevalEvaluator
    │                             │                      │                      │
    │  remote_config:             │                      │                      │
    │   judge_model=gpt-4o-mini   │                      │                      │
    │   judge_base_url=https://.. │                      │                      │
    │────────────────────────────▶│                      │                      │
    │                             │  update({            │                      │
    │                             │   judge_model: ...,  │                      │
    │                             │   judge_base_url:..})│                      │
    │                             │─────────────────────▶│                      │
    │                             │                      │                      │
    │                             │                      │  (next eval call)    │
    │                             │                      │◀─────────────────────│
    │                             │                      │  get(judge_model)    │
    │                             │                      │  get(judge_base_url) │
    │                             │                      │─────────────────────▶│
    │                             │                      │                      │
    │                             │                      │     Uses gpt-4o-mini │
    │                             │                      │     for this eval    │
```

---

## Risk and Mitigation

> **Update (2026-03-05):** Upstream PR #3635 has been **merged** into
> `opentelemetry-python-contrib`. The `opentelemetry-opamp-client` package
> is now available on `main`. The "not merged" risk is retired.

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Upstream `_opamp` module is private (`opentelemetry._opamp`) | API may change without semver guarantees between releases | Pin `opentelemetry-opamp-client` to a known-good version; adapter layer isolates the rest of our code from upstream API changes. Only `handler.py` in the adapter touches the upstream API |
| Upstream `message_handler` signature changes | Adapter callback breaks | Thin adapter: only `handler.py` calls upstream; single file to update |
| Upstream `decode_remote_config` only supports JSON content types | Non-JSON configs from server silently ignored | Document JSON requirement for OpAMP server operators; add content-type validation in adapter with clear error logging |
| Thread safety in config propagation | Race conditions between OpAMP daemon thread and app threads | `ConfigStore` uses `RLock`; observers are collected under lock but invoked outside it to prevent deadlocks; no direct env var mutation from OpAMP thread |
| In-flight evaluations during reconfiguration | Evaluations may use stale evaluator instances or fail | `Manager.reconfigure()` atomically swaps plans and evaluators; in-flight items already dequeued continue with previous evaluator set |
| OpAMP server pushes invalid config | Agent misbehaves or crashes | `key_mapper.py` validates and clamps all values (e.g., sample rate to 0.0-1.0); invalid configs logged and rejected; last-known-good config preserved |
| Judge model hot-swap during active evaluation | LLM call uses wrong model mid-evaluation | Judge model env vars are propagated atomically; reconfiguration only affects evaluations picked up after the swap |
| `requests` library removed from upstream transport | HTTP transport breaks | `HttpTransport` is an ABC; we can provide an alternative implementation (e.g., `urllib3`) if upstream drops `requests` |
| `opentelemetry-opamp-client` not yet published to PyPI | Cannot declare it as a pip dependency | Install from git or vendor locally during development; switch to PyPI dependency once published |
| OpAMP connection loss or server unavailability | Config updates stop arriving | Upstream `OpAMPAgent` has exponential backoff and retry built in; last-known-good config in `ConfigStore` remains active; no functionality lost, only dynamic updates paused |
