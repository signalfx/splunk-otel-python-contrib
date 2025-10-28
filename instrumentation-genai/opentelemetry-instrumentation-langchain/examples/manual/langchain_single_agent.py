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
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

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
Trace ID: 4a0203cac3bcf7dc40960d938c257278
└── Span ID: 64f3b947ba47d433 (Parent: none) - Name: invoke_agent weather-agent [op:invoke_agent] (Type: span)
    ├── Log: gen_ai.client.agent.operation.details (Type: log)
    ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    ├── Metric: gen_ai.agent.duration [op:invoke_agent] (Type: metric)
    ├── Metric: gen_ai.evaluation.bias [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.relevance [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
    ├── Metric: gen_ai.evaluation.toxicity [op:evaluation] (Type: metric)
    ├── Span ID: 072ebe1698abf8b5 (Parent: 64f3b947ba47d433) - Name: gen_ai.step model (Type: span)
    │   ├── Span ID: 8cdd0c6c54488dae (Parent: 072ebe1698abf8b5) - Name: chat ChatOpenAI [op:chat] (Type: span)
    │   │   ├── Log: gen_ai.client.inference.operation.details [op:chat] (Type: log)
    │   │   ├── Log: gen_ai.evaluation.results [op:data_evaluation_results] (Type: log)
    │   │   ├── Metric: gen_ai.evaluation.hallucination [op:evaluation] (Type: metric)
    │   │   └── Metric: gen_ai.evaluation.sentiment [op:evaluation] (Type: metric)
    │   └── Span ID: 46d14eaab07d930c (Parent: 072ebe1698abf8b5) - Name: gen_ai.step model_to_tools (Type: span)
    ├── Span ID: 8d4cf6fa661400e1 (Parent: 64f3b947ba47d433) - Name: gen_ai.step tools (Type: span)
    │   ├── Metric: gen_ai.step.duration (Type: metric)
    │   ├── Span ID: 6cb4be88f8e83162 (Parent: 8d4cf6fa661400e1) - Name: tool get_weather [op:execute_tool] (Type: span)
    │   │   └── Metric: gen_ai.client.operation.duration [op:execute_tool] (Type: metric)
    │   └── Span ID: 99799bc83ead5255 (Parent: 8d4cf6fa661400e1) - Name: gen_ai.step tools_to_model (Type: span)
    │       └── Metric: gen_ai.step.duration (Type: metric)
    └── Span ID: af9231938e7f64b8 (Parent: 64f3b947ba47d433) - Name: gen_ai.step model (Type: span)
        ├── Metric: gen_ai.step.duration (Type: metric)
        ├── Span ID: 72769969acd490ce (Parent: af9231938e7f64b8) - Name: chat ChatOpenAI [op:chat] (Type: span)
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
        └── Span ID: 8442104e9276b884 (Parent: af9231938e7f64b8) - Name: gen_ai.step model_to_tools (Type: span)
            └── Metric: gen_ai.step.duration (Type: metric)

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
).with_config({
    "run_name": "weather-agent",
    "tags": ["agent:weather", "agent"],
    "metadata": {"agent_name": "weather-agent", "agent_role": "orchestrator"}
})

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]},
    {"session_id": "12345"},
)

# sleep for 150s to allow evals to finish 
import time
time.sleep(150)