# In-House Instrumentation Example: Multi-Agent Travel Planner

This directory shows how to manually instrument an in‑house (LangGraph / LangChain‑based) multi‑agent workflow using the structured GenAI types provided by `opentelemetry.util.genai.types`.

The core types:

* `Workflow` – high‑level orchestration span (end‑to‑end request lifecycle).
* `AgentInvocation` – one logical agent or tool reasoning step.
* `LLMInvocation` – a single model call (chat / completion / embeddings).
* `InputMessage` / `OutputMessage` – structured message parts (role + list of parts). Each part can be a `Text`, image, etc.

Benefits of using these types instead of ad‑hoc span attributes:

1. Consistency – every model call captures inputs, outputs, tokens the same way.
2. Extensibility – evaluation / replay / redaction layers can rely on stable data shapes.
3. Safety – avoids leaking PII by keeping messages as typed parts you can filter before export.
4. Metrics – token counts populate standard semantic fields without manual key guessing.

---

## Minimal LLMInvocation Example (Single OpenAI Chat Call – Direct OpenAI Client)

```python
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    Workflow,
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
)
from openai import OpenAI

# Requires: pip install openai ; environment variable OPENAI_API_KEY set.

handler = get_telemetry_handler()

workflow = Workflow(
    name="demo_workflow",
    workflow_type="single_call",
    description="One-off chat completion",
    initial_input="Hello, can you summarise OpenTelemetry?",
)
handler.start_workflow(workflow)

llm_invocation = LLMInvocation(
    request_model="gpt-4o-mini",  # model identifier
    operation="chat",
    input_messages=[
        InputMessage(role="system", parts=[Text(content="You are a concise assistant.")]),
        InputMessage(role="user", parts=[Text(content=workflow.initial_input or "")]),
    ],
)
llm_invocation.provider = "openai"
llm_invocation.framework = "native-client"
handler.start_llm(llm_invocation)

# Convert InputMessages to OpenAI API format (list of {role, content} dicts)
openai_messages = [
    {"role": m.role, "content": "".join(part.content for part in m.parts if hasattr(part, "content"))}
    for m in llm_invocation.input_messages
]

client = OpenAI()
response = client.chat.completions.create(
    model=llm_invocation.request_model,
    messages=openai_messages,
    temperature=0.2,
)

# Extract assistant answer
choice = response.choices[0]
assistant_text = choice.message.content

llm_invocation.output_messages = [
    OutputMessage(role="assistant", parts=[Text(content=assistant_text)], finish_reason=choice.finish_reason or "stop")
]

# Token usage (OpenAI returns usage.prompt_tokens / usage.completion_tokens / usage.total_tokens)
if response.usage:
    llm_invocation.input_tokens = response.usage.prompt_tokens
    llm_invocation.output_tokens = response.usage.completion_tokens

handler.stop_llm(llm_invocation)

workflow.final_output = assistant_text
handler.stop_workflow(workflow)
```

Key points:

* All user/system inputs are captured up front (`input_messages`).
* The model response becomes `output_messages` (list for multi‑turn or tool streaming scenarios).
* Token counts live on the invocation object – downstream metrics aggregators don’t need to parse raw attributes.

---

## AgentInvocation + LLMInvocation (Typical Pattern – Direct OpenAI Client)

When an agent first reasons about a task (planning, tool selection) you can represent that with `AgentInvocation`. Inside the agent you usually trigger one or more `LLMInvocation`s.

```python
from opentelemetry.util.genai.types import (
    Workflow,
    AgentInvocation,
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from openai import OpenAI

handler = get_telemetry_handler()
workflow = Workflow(name="agent_demo", workflow_type="planner", initial_input="Plan a 2-day trip to Rome")
handler.start_workflow(workflow)

agent = AgentInvocation(
    name="trip_planner",
    agent_type="planner",
    model="gpt-4o-mini",
    system_instructions="You plan concise city itineraries",
    input_context=workflow.initial_input,
)
handler.start_agent(agent)

llm_call = LLMInvocation(
    request_model="gpt-4o-mini",
    operation="chat",
    input_messages=[
        InputMessage(role="system", parts=[Text(content="You provide day-by-day plans.")]),
        InputMessage(role="user", parts=[Text(content="Plan a 2-day trip to Rome focusing on food and history.")]),
    ],
)
llm_call.provider = "openai"
llm_call.framework = "native-client"
handler.start_llm(llm_call)

client = OpenAI()
openai_messages = [
    {"role": m.role, "content": "".join(p.content for p in m.parts if hasattr(p, "content"))}
    for m in llm_call.input_messages
]
response = client.chat.completions.create(
    model=llm_call.request_model,
    messages=openai_messages,
    temperature=0.3,
)

choice = response.choices[0]
assistant_text = choice.message.content
llm_call.output_messages = [
    OutputMessage(role="assistant", parts=[Text(content=assistant_text)], finish_reason=choice.finish_reason or "stop")
]
if response.usage:
    llm_call.input_tokens = response.usage.prompt_tokens
    llm_call.output_tokens = response.usage.completion_tokens

agent.output_result = assistant_text
handler.stop_llm(llm_call)
handler.stop_agent(agent)
workflow.final_output = assistant_text
handler.stop_workflow(workflow)
```

Why this structure helps:

* Multiple `LLMInvocation`s inside one agent (tool lookups, reasoning, synthesis) stay grouped beneath the agent span.
* You can decorate the agent span with evaluation signals later (e.g. quality score) without touching core LLM spans.
* Redaction / filtering can operate at message part granularity before export.

---

## Helper Strategy (Token + Output Auto-Population)

In the travel planner example we use a helper to:

1. Create `output_messages` if the node hasn’t set them yet.
2. Extract token usage from LangChain’s `usage_metadata` or `response_metadata.token_usage`.

Pattern:

```python
_apply_llm_response_metadata(response_message, llm_invocation)
```

Call this immediately after each model invocation (direct OpenAI response object), then stop the LLM span.

---

## Adding Evaluations Later

Because inputs/outputs are normalized:

* You can iterate over finished `LLMInvocation`s and feed them to an evaluation agent (latency, toxicity, factuality).
* Link evaluation spans as children or siblings referencing the `llm_invocation_id`.

---

## Minimal Lifecycle Checklist

1. Start `Workflow` (once per external request).
2. For each logical reasoning component: start `AgentInvocation`.
3. Inside agent: start one or more `LLMInvocation` spans.
4. Populate `input_messages` before the call; populate `output_messages` + tokens right after.
5. Stop spans in reverse order (LLM → Agent → Workflow).

---

## Troubleshooting

* Missing tokens? Ensure your client/library actually returns usage metadata; not all providers do.
* Dropped messages? Confirm you set both `input_messages` and `output_messages` *before* stopping the LLM span.
* Need streaming? Append incremental `OutputMessage` parts as they arrive; finalise with a finish_reason of `stop` or `length`.

---
