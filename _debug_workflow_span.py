"""Debug script to check what span name LangChain creates for workflows."""

import os

os.environ["OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS"] = "none"
os.environ["OTEL_INSTRUMENTATION_GENAI_EMITTERS"] = "span_metric_event"

from uuid import uuid4

from opentelemetry.instrumentation.langchain.callback_handler import (
    LangchainCallbackHandler,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import get_telemetry_handler

exp = InMemorySpanExporter()
tp = TracerProvider()
tp.add_span_processor(SimpleSpanProcessor(exp))
mp = MeterProvider()

if hasattr(get_telemetry_handler, "_default_handler"):
    delattr(get_telemetry_handler, "_default_handler")
handler = get_telemetry_handler(tracer_provider=tp, meter_provider=mp)
cb = LangchainCallbackHandler(telemetry_handler=handler)

root_id = uuid4()
cb.on_chain_start(
    serialized={"name": "LangGraphWorkflow"},
    inputs={"input": "start"},
    run_id=root_id,
    tags=[],
    metadata={"langgraph_node": "__start__"},
)
cb.on_chain_end(outputs={"output": "done"}, run_id=root_id)

spans = exp.get_finished_spans()
for s in spans:
    print(f"Span: {s.name}, attrs: {dict(s.attributes or {})}")

if not spans:
    print("NO SPANS FOUND")
