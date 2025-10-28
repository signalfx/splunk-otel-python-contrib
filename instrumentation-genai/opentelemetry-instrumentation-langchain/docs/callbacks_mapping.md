# OpenTelemetry LangChain Instrumentation – Callback → GenAI Type Mapping

This document summarizes how the LangChain callback handler in `instrumentation-genai/opentelemetry-instrumentation-langchain-dev` maps runtime events to the GenAI telemetry types defined in `util/opentelemetry-util-genai-dev/src/opentelemetry/util/genai/types.py`. The content reflects the Phase 1 handler already present on this branch.

## Callback → Entity Matrix

| Callback | GenAI entity | Notes |
|----------|--------------|-------|
| `on_chain_start` (no parent) | `AgentInvocation` when `tags`/`metadata` resolve an `agent_name` (e.g. tag `agent:foo`, `metadata.agent_name`); otherwise `Workflow`. | Sets `framework="langchain"`, stores serialized inputs (`input_context` or `initial_input`), adopts `metadata.agent_type`, `metadata.model_name`, `metadata.system` when present. |
| `on_chain_start` (has parent) | `AgentInvocation` when `tags`/`metadata` introduce a *new* `agent_name` (different to the nearest ancestor agent); else `ToolCall` when tool hints are present; otherwise `Step` with `step_type="chain"`. | Agent promotion keeps orchestration spans aligned with agent boundaries. Tool paths reuse an existing `ToolCall` (if `on_tool_start` fired first), refresh `arguments`, and stash JSON in `attributes["tool.arguments"]`. Non-tool paths create a `Step` and serialize inputs into `step.input_data`. |
| `on_chat_model_start` / `on_llm_start` | `LLMInvocation`. | Prompts/messages become `InputMessage` parts, model name resolved from serialized payload → metadata → kwargs; agent context propagated from parent entity; `on_llm_start` simply calls the chat handler and overwrites `operation="generate_text"`. |
| `on_llm_end` | — | Fills `output_messages`, extracts token counts from `response.llm_output.usage`, then stops the invocation. |
| `on_tool_start` | `ToolCall`. | Creates (or updates) a `ToolCall` with normalized name/id, serializes `inputs`/`input_str`, records agent context when parent is an `AgentInvocation`, sets `attributes["tool.arguments"]`. |
| `on_tool_end` | — | Serializes tool output into `attributes["tool.response"]` and stops the call. |
| `on_chain_end` | — | Resolves the registered entity for `run_id` (`Workflow`, `AgentInvocation`, `Step`, or `ToolCall`), copies serialized outputs to the matching field, then stops it. |
| Any `*_error` | — | Invokes `fail_by_run_id(run_id, GenAIError(message=str(error), type=type(error)))`; the util layer handles span status and cleanup. |

Common behaviours:

- Tags are preserved on every entity via `attributes["tags"]`.
- `_serialize` prefers JSON (UTF-8 preserved) with `str()` fallback; un-serializable payloads become `None`.
- Agent-aware parenting walks up to the nearest `AgentInvocation`, ensuring `Step`, `ToolCall`, and `LLMInvocation` entities inherit `agent_name` / `agent_id`. If callbacks supply a different `agent_name`, the handler promotes that chain to a new `AgentInvocation` span before continuing.


## Sample Trace (`examples/manual/t.py`)

```text
Trace ID: 879de17ad8b2ce0565d081fcceea5238
└── Span ID: eed435b2128ed02a  - invoke_agent weather-agent [op:invoke_agent]
    ├── Span ID: ba3fca28e7b484da  - gen_ai.step model
    │   ├── Span ID: 9567d0b878dce12b  - chat ChatOpenAI [op:chat]
    │   └── Span ID: 9ef4fd28e1fac76f  - gen_ai.step model_to_tools
    ├── Span ID: 1c4c2c3561a6bdaa  - gen_ai.step tools
    │   ├── Span ID: d9715ba2dc8026fe  - tool get_weather [op:execute_tool]
    │   └── Span ID: da16fac4673f3da1  - gen_ai.step tools_to_model
    └── Span ID: f5330649bb2b3ba0  - gen_ai.step model
        ├── Span ID: a85348539e216d94  - chat ChatOpenAI [op:chat]
        └── Span ID: c0ecb30e07fe5291  - gen_ai.step model_to_tools
```

Highlights:

- A single agent span roots the trace; chain steps and tool calls remain within the same trace.
- Tool execution is captured twice: once as the orchestration `Step` (`gen_ai.step tools`) and once as the `ToolCall` span (`tool get_weather`).
- Follow-up model/tool orchestration stays under the agent root through `parent_run_id` propagation.

## Multi-Agent Trace (`examples`)
```text
Trace ID: 2dfc14dffa52abf855a1fdcd9c52c83a
└── Span ID: c29f8be9111e89ee (Parent: none) - Name: invoke_agent travel_multi_agent_planner [op:invoke_agent] (Type: span)
    └── Span ID: 61b885b3526c333d (Parent: c29f8be9111e89ee) - Name: gen_ai.workflow LangGraph (Type: span)
        ├── Span ID: f9f1ec44e0e2bf24 (Parent: 61b885b3526c333d) - Name: gen_ai.step __start__ (Type: span)
        │   └── Span ID: a1cf31eed3742bb4 (Parent: f9f1ec44e0e2bf24) - Name: gen_ai.step should_continue (Type: span)
        ├── Span ID: fcaf50aa92ac0474 (Parent: 61b885b3526c333d) - Name: gen_ai.step coordinator (Type: span)
        │   ├── Span ID: 5979cb43ec683cd4 (Parent: fcaf50aa92ac0474) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   └── Span ID: 8bb96ae29bf02d9d (Parent: fcaf50aa92ac0474) - Name: gen_ai.step should_continue (Type: span)
        ├── Span ID: 8790453f9bd9e789 (Parent: 61b885b3526c333d) - Name: gen_ai.step flight_specialist (Type: span)
        │   ├── Span ID: 60105d1e53860b92 (Parent: 8790453f9bd9e789) - Name: invoke_agent flight_specialist [op:invoke_agent] (Type: span)
        │   │   ├── Span ID: 46403ca00ba8f07b (Parent: 60105d1e53860b92) - Name: gen_ai.step model (Type: span)
        │   │   │   ├── Span ID: c9cc5c5d4bf74049 (Parent: 46403ca00ba8f07b) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │   │   └── Span ID: c43b4f51f5a0d27a (Parent: 46403ca00ba8f07b) - Name: gen_ai.step model_to_tools (Type: span)
        │   │   ├── Span ID: 03d205fa95afa273 (Parent: 60105d1e53860b92) - Name: gen_ai.step tools (Type: span)
        │   │   │   ├── Span ID: e35edaff2e6d8ddc (Parent: 03d205fa95afa273) - Name: tool mock_search_flights [op:execute_tool] (Type: span)
        │   │   │   └── Span ID: 948b226d4f78b8c6 (Parent: 03d205fa95afa273) - Name: gen_ai.step tools_to_model (Type: span)
        │   │   └── Span ID: 309204eac5fd291b (Parent: 60105d1e53860b92) - Name: gen_ai.step model (Type: span)
        │   │       ├── Span ID: 2c2c2a49c6fd07c2 (Parent: 309204eac5fd291b) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │       └── Span ID: 61314a8080d85f32 (Parent: 309204eac5fd291b) - Name: gen_ai.step model_to_tools (Type: span)
        │   └── Span ID: 3ed967fd966a3808 (Parent: 8790453f9bd9e789) - Name: gen_ai.step should_continue (Type: span)
        ├── Span ID: c84e59a40caa4e67 (Parent: 61b885b3526c333d) - Name: gen_ai.step hotel_specialist (Type: span)
        │   ├── Span ID: db58c106cfda4a30 (Parent: c84e59a40caa4e67) - Name: invoke_agent hotel_specialist [op:invoke_agent] (Type: span)
        │   │   ├── Span ID: e3f10a50c5fee140 (Parent: db58c106cfda4a30) - Name: gen_ai.step model (Type: span)
        │   │   │   ├── Span ID: 267410f812ad44f9 (Parent: e3f10a50c5fee140) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │   │   └── Span ID: deb4fec45578e183 (Parent: e3f10a50c5fee140) - Name: gen_ai.step model_to_tools (Type: span)
        │   │   ├── Span ID: 252f02c4edea2331 (Parent: db58c106cfda4a30) - Name: gen_ai.step tools (Type: span)
        │   │   │   ├── Span ID: 5eea1621cd3231c3 (Parent: 252f02c4edea2331) - Name: tool mock_search_hotels [op:execute_tool] (Type: span)
        │   │   │   └── Span ID: 4ceac9860e0e2755 (Parent: 252f02c4edea2331) - Name: gen_ai.step tools_to_model (Type: span)
        │   │   └── Span ID: 79fbf964f8700762 (Parent: db58c106cfda4a30) - Name: gen_ai.step model (Type: span)
        │   │       ├── Span ID: 5aa6f47d8e4ebfd3 (Parent: 79fbf964f8700762) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │       └── Span ID: fdde80a693217b35 (Parent: 79fbf964f8700762) - Name: gen_ai.step model_to_tools (Type: span)
        │   └── Span ID: 1bc626b6aa767596 (Parent: c84e59a40caa4e67) - Name: gen_ai.step should_continue (Type: span)
        ├── Span ID: 4e15833d5786e97b (Parent: 61b885b3526c333d) - Name: gen_ai.step activity_specialist (Type: span)
        │   ├── Span ID: 798284f44f1059cd (Parent: 4e15833d5786e97b) - Name: invoke_agent activity_specialist [op:invoke_agent] (Type: span)
        │   │   ├── Span ID: f9ccbfdb645b82e5 (Parent: 798284f44f1059cd) - Name: gen_ai.step model (Type: span)
        │   │   │   ├── Span ID: e204ff16421bcccb (Parent: f9ccbfdb645b82e5) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │   │   └── Span ID: 19d4a6091bd97877 (Parent: f9ccbfdb645b82e5) - Name: gen_ai.step model_to_tools (Type: span)
        │   │   ├── Span ID: 7516668a7f89e8eb (Parent: 798284f44f1059cd) - Name: gen_ai.step tools (Type: span)
        │   │   │   ├── Span ID: 1cecb9ac1bffd603 (Parent: 7516668a7f89e8eb) - Name: tool mock_search_activities [op:execute_tool] (Type: span)
        │   │   │   └── Span ID: 92fe78ef9abb5edb (Parent: 7516668a7f89e8eb) - Name: gen_ai.step tools_to_model (Type: span)
        │   │   └── Span ID: 3fcdf744189e8d41 (Parent: 798284f44f1059cd) - Name: gen_ai.step model (Type: span)
        │   │       ├── Span ID: 1ab21415cd276763 (Parent: 3fcdf744189e8d41) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   │       └── Span ID: b7c81d5b6006acfa (Parent: 3fcdf744189e8d41) - Name: gen_ai.step model_to_tools (Type: span)
        │   └── Span ID: 0906d16730ed8e4e (Parent: 4e15833d5786e97b) - Name: gen_ai.step should_continue (Type: span)
        └── Span ID: 7418fabb26d5cce6 (Parent: 61b885b3526c333d) - Name: gen_ai.step plan_synthesizer (Type: span)
            ├── Span ID: c3203b76db83439c (Parent: 7418fabb26d5cce6) - Name: chat ChatOpenAI [op:chat] (Type: span)
            └── Span ID: 7fa97fa4a67e4eb0 (Parent: 7418fabb26d5cce6) - Name: gen_ai.step should_continue (Type: span)
```

## Lifecycle Cheat Sheet

```text
on_chain_start (root)
    ├─ agent metadata → start_agent(AgentInvocation)
    └─ otherwise     → start_workflow(Workflow)
        │
        └─ on_chain_start (child)
             ├─ tool metadata → start_tool_call(ToolCall)
             │       └─ on_tool_end → stop_tool_call
             └─ default       → start_step(Step)
                     ├─ on_chat_model_start / on_llm_start → start_llm
                     ├─ on_llm_end                          → stop_llm
                     └─ on_chain_end                        → stop_step
        └─ on_chain_end → stop_agent / stop_workflow

Errors (any callback) → fail_by_run_id → TelemetryHandler.fail_* → span status = ERROR
```

Keep this document aligned with the handler whenever new callbacks or entity types gain support.

