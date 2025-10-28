# Traceloop -> GenAI Semantic Convention Translator Emitter

This optional emitter promotes legacy `traceloop.*` attributes attached to an `LLMInvocation` into
Semantic Convention (or forward-looking custom `gen_ai.*`) attributes **before** the standard
Semantic Convention span emitter runs. It does **not** create its own span.

## Why Use It?
If you have upstream code (or the Traceloop compat emitter) producing `traceloop.*` keys but you
want downstream dashboards/tools to rely on GenAI semantic conventions, enabling this translator
lets you transition without rewriting upstream code immediately.

## What It Does
At `on_start` of an `LLMInvocation` it scans `invocation.attributes` for keys beginning with
`traceloop.` and adds corresponding keys:

| Traceloop Key (prefixed or raw) | Added Key                 | Notes |
|---------------------------------|---------------------------|-------|
| `traceloop.workflow.name` | `gen_ai.workflow.name`  | Custom (not yet in spec) |
| `traceloop.entity.name`     | `gen_ai.agent.name`     | Approximates entity as agent name |
| `traceloop.entity.path`     | `gen_ai.workflow.path`  | Custom placeholder |
| `traceloop.callback.name`  | `gen_ai.callback.name`  | Also sets `gen_ai.operation.source` if absent |
| `traceloop.callback.id`     | `gen_ai.callback.id`    | Custom |
| `traceloop.entity.input`   | `gen_ai.input.messages` | Serialized form already present |
| `traceloop.entity.output` | `gen_ai.output.messages`| Serialized form already present |

Existing `gen_ai.*` keys are never overwritten.

## Installing
Fast path (no entry point needed):

```bash
   pip install opentelemetry-util-genai-traceloop-translator
```



## Example
Minimal Python snippet (assuming emitters are loaded via entry points and the translator is installed):

```python
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text

inv = LLMInvocation(
    request_model="gpt-4",
    input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
    attributes={
        "traceloop.entity.name": "ChatLLM",
        "traceloop.workflow.name": "user_flow",
        "traceloop.callback.name": "root_chain",
        "traceloop.entity.input": "[{'role':'user','content':'Hello'}]",
    },
)
handler = get_telemetry_handler()
handler.start_llm(inv)
inv.output_messages = [OutputMessage(role="assistant", parts=[Text("Hi")], finish_reason="stop")]
handler.stop_llm(inv)
# Result: final semantic span contains gen_ai.agent.name, gen_ai.workflow.name, gen_ai.input.messages, etc.
```
