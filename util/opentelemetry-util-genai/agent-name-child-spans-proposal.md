# Proposal: `gen_ai.agent.name` on GenAI child spans and client metrics

> **Intent:** Contribution draft for [open-telemetry/semantic-conventions](https://github.com/open-telemetry/semantic-conventions).  
> **Scope:** `gen_ai.agent.name` only — **`gen_ai.agent.id` is explicitly out of scope.**

---

## 1. Motivation / Problem statement

Multi-agent and orchestrated applications emit many **inference**, **embeddings**, **retrieval**, and **execute_tool** spans that share the same **`gen_ai.request.model`** or **`gen_ai.tool.name`**. Those attributes alone do not identify **which logical agent** (e.g. planner vs retriever) initiated the operation.

Today, operators must **reconstruct** agent context by **walking the trace** (e.g. from an `invoke_agent` parent). That is fragile when:

- Traces are **incomplete**, **sampled**, or spans are analyzed **without** full parent chains.
- Backends need **simple filters** (`operation` + `model` + **agent**) without graph joins on every query.

**Metrics** for `gen_ai.client.token.usage` and `gen_ai.client.operation.duration` are similarly hard to break down **by agent** without a standard attribute, which blocks **cost**, **latency**, and **SLO** views per agent.

---

## 2. Goals

- Standardize **`gen_ai.agent.name`** on **inference**, **embeddings**, **retrieval**, and **execute_tool** **client** spans when the operation is performed **on behalf of a named agent**.
- Add **`gen_ai.agent.name`** as a **documented** dimension on **GenAI client metrics** where it improves breakdown without mandating high cardinality.
- Keep **`gen_ai.agent.name`** as a **low-cardinality**, **logical** agent label (product/agent role), not a per-run identifier.

---

## 3. Non-goals

- **`gen_ai.agent.id`** or any **instance-level** agent identity on these spans or metrics (explicitly out of scope).
- Changing **required** attribute sets in a way that **breaks** existing instrumentations (prefer **recommended** / **opt-in** for metrics).

---

## 4. Proposed solution

### 4.1 Semantic meaning

**`gen_ai.agent.name`** on a **child** span or metric record means:

> The **logical name** of the agent **on whose behalf** this inference, embedding, retrieval, or tool execution was performed.

It **SHOULD** align with the name used when that agent is represented by an **`invoke_agent`** (or equivalent) span in the same system, when such a span exists.

### 4.2 Span convention changes (`gen-ai-spans.md`)

For each of the following sections, **add** `gen_ai.agent.name` to the span attribute table:

| Section | Span kinds / notes |
|--------|---------------------|
| Inference | e.g. `chat`, `generate_content`, `text_completion`, … |
| Embeddings | `embeddings` |
| Retrievals | `retrieval` |
| Execute tool | `execute_tool` |

**Suggested requirement level:** **Recommended** — when the instrumentation **knows** the agent name (typical for agent frameworks / wrappers). Omitted when there is **no** agent concept (raw model client).

**Documentation notes (normative guidance):**

- **MUST NOT** use this attribute for **end-user IDs**, **request IDs**, or other **unbounded** values.
- Instrumentations **SHOULD** use a **small, stable** set of names (e.g. `billing_support`, `research_agent`).

### 4.3 Metric convention changes (`gen-ai-metrics.md`)

Add **`gen_ai.agent.name`** to metric attribute tables where the operation can be tied to an agent, for example:

| Metric | Suggested requirement |
|--------|------------------------|
| `gen_ai.client.token.usage` | Recommended when available |
| `gen_ai.client.operation.duration` | Recommended when available |
| *(Optional)* `gen_ai.client.operation.time_to_first_chunk` | Recommended when available |
| *(Optional)* `gen_ai.client.operation.time_per_output_chunk` | Recommended when available |

**Guidance:** Same low-cardinality rules as spans; implementations **MAY** omit when no agent context exists.

---

## 5. Use cases / rationale

### 5.1 Spans

- **Filtering and grouping** in trace UIs without inferring parent `invoke_agent`.
- **Disambiguation** when the same **model** or **tool** is used by **different** agents.
- **Attribution** when spans are stored or analyzed **without** full trace context.

### 5.2 Metrics

- **Token and cost** breakdown by agent.
- **Latency and error** SLOs **per agent** for the same `gen_ai.operation.name` and model.
- **Alerting** scoped to a specific agent’s behavior.

---

## 6. Backward compatibility

- **Additive** only: new **recommended** (or **opt-in** for metrics, if SIG prefers) attributes/dimensions.
- Respect existing GenAI **stability and opt-in** policy for emitting **latest experimental** vs legacy behavior.

---

## 7. Alternatives considered

| Alternative | Why not chosen |
|-------------|----------------|
| Rely only on **trace parent** (`invoke_agent`) | Fails for partial traces, sampling, and simple backend queries. |
| Use **custom** / vendor-specific attributes | Prevents **portable** dashboards and cross-vendor tooling. |
| Add **`gen_ai.agent.id`** on metrics | High **cardinality** risk; explicitly excluded from this proposal. |
| **Required** `gen_ai.agent.name` on all child spans | Breaks **non-agent** model client usage. |

---

## 8. Open questions

1. **Nested agents:** Should the spec say **“nearest owning agent”** vs **“root workflow agent”** when multiple agents nest? (Pick one default; allow instrumentation notes.)
2. **Metrics requirement level:** **Recommended** vs **Opt-in** for `gen_ai.client.*` metrics—SIG preference for default cardinality.
3. **Streaming metrics:** Include **`gen_ai.agent.name`** on **time_to_first_chunk** / **time_per_output_chunk** in v1 of the change or follow-up PR?

---

## 9. Specification / implementation checklist

- [ ] Update **`model/`** YAML for affected span and metric definitions.
- [ ] Regenerate **`docs/gen-ai/gen-ai-spans.md`** and **`docs/gen-ai/gen-ai-metrics.md`**.
- [ ] **CHANGELOG** entry under GenAI.
- [ ] Optional: examples in **non-normative** docs showing agent-attributed chat + tool spans.

---

## 10. References

- [OpenTelemetry Semantic Conventions repository](https://github.com/open-telemetry/semantic-conventions)
- [GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)
- [GenAI metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md)
- [Contributing](https://github.com/open-telemetry/semantic-conventions/blob/main/CONTRIBUTING.md)
