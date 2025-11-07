import time

from langchain.agents import create_agent
from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing/metrics/logging once per process so exported data goes to OTLP.
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

demo_tracer = trace.get_tracer("instrumentation.langchain.demo")

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
_events.set_event_logger_provider(EventLoggerProvider())

instrumentor = LangchainInstrumentor()
instrumentor.instrument()

"""
A simple LangChain single-agent example with LangChain instrumentaiton using OpenTelemetry GenAI Utils with evals enabled.

Example telemetry expected: 
Trace ID: d6d3550630667b15dc20c521ea3abd2a
└── Span ID: 3363678cbe837b99 (Parent: none) - Name: invoke_agent weather-agent [op:chat] (Type: span)
    ├── Log: gen_ai.client.agent.operation.details (Type: log)
    ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    ├── Metric: gen_ai.agent.duration [op:invoke_agent] (Type: metric)
    ├── Metric: gen_ai.evaluation.bias [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.relevance [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.toxicity [op:evaluation] (Type: metric)
    ├── Span ID: a6679f427633fcb9 (Parent: 3363678cbe837b99) - Name: gen_ai.step model (Type: span)
    │   ├── Span ID: 727c71391cb03753 (Parent: a6679f427633fcb9) - Name: chat ChatOpenAI [op:chat] (Type: span)
    │   │   ├── Log: gen_ai.client.inference.operation.details [op:chat] (Type: log)
    │   │   ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    │   │   ├── Metric: gen_ai.client.operation.duration [op:chat] (Type: metric)
    │   │   ├── Metric: gen_ai.client.token.usage (output) [op:chat] (Type: metric)
    │   │   ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
    │   │   └── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
    │   └── Span ID: 990a8858820e136e (Parent: a6679f427633fcb9) - Name: gen_ai.step model_to_tools (Type: span)
    ├── Span ID: 73b3a8ef46b36a93 (Parent: 3363678cbe837b99) - Name: gen_ai.step tools [op:chat] (Type: span)
    │   ├── Span ID: b14699dc6e4d265d (Parent: 73b3a8ef46b36a93) - Name: tool get_weather [op:execute_tool] (Type: span)
    │   │   └── Metric: gen_ai.client.operation.duration [op:execute_tool] (Type: metric)
    │   └── Span ID: 2ff06d8b3d24aa60 (Parent: 73b3a8ef46b36a93) - Name: gen_ai.step tools_to_model (Type: span)
    └── Span ID: 91ff632289379944 (Parent: 3363678cbe837b99) - Name: gen_ai.step model (Type: span)
        ├── Span ID: 104ef8823199fdd1 (Parent: 91ff632289379944) - Name: chat ChatOpenAI [op:chat] (Type: span)
        │   ├── Log: gen_ai.client.inference.operation.details [op:chat] (Type: log)
        │   ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
        │   ├── Metric: gen_ai.client.operation.duration [op:chat] (Type: metric)
        │   ├── Metric: gen_ai.client.token.usage (input) [op:chat] (Type: metric)
        │   ├── Metric: gen_ai.client.token.usage (output) [op:chat] (Type: metric)
        │   ├── Metric: gen_ai.evaluation.bias [op:evaluation] (Type: metric)
        │   ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
        │   ├── Metric: gen_ai.evaluation.relevance [op:evaluation] (Type: metric)
        │   ├── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
        │   └── Metric: gen_ai.evaluation.toxicity [op:evaluation] (Type: metric)
        └── Span ID: 210e81dd011c2cb7 (Parent: 91ff632289379944) - Name: gen_ai.step model_to_tools (Type: span)

vscode launch configuration to run this example:
```
{
      "name": "langchain_single_agent.py",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/instrumentation-genai/opentelemetry-instrumentation-langchain-dev/examples/manual/langchain_single_agent.py",
      "cwd": "${workspaceFolder}/instrumentation-genai/opentelemetry-instrumentation-langchain-dev/examples/manual/",
      "python": "${workspaceFolder}/.venv/bin/python",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": "DELTA",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED": "true",
        "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment=o11y-for-ai-dev-sergey,scenario=splunk_eval",
        "OTEL_SERVICE_NAME": "demo-app-util-genai-dev",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE": "SPAN_AND_EVENT",
        "OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION": "true",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS": "span_metric_event,splunk",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS_EVALUATION": "replace-category:SplunkEvaluationResults",
        "OTEL_GENAI_EVAL_DEBUG_SKIPS": "true",
        "OTEL_GENAI_EVAL_DEBUG_EACH": "true",
        "OTEL_INSTRUMENTATION_LANGCHAIN_DEBUG": "true",
        // "OPENAI_API_KEY": "your-openai-key"
      }
    }
```
"""


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    name="weather-agent",
    model="openai:gpt-5-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    debug=True,
).with_config(
    {
        "run_name": "weather-agent",
        "tags": ["agent:weather", "agent"],
        "metadata": {"agent_name": "weather-agent", "agent_role": "orchestrator"},
    }
)

# Run the agent
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ]
    },
    {"session_id": "12345"},
)

# sleep for 150s to allow evals to finish
time.sleep(150)
