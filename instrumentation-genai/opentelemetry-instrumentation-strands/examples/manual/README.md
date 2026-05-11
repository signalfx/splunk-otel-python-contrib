# Strands Instrumentation Example

This example demonstrates OpenTelemetry telemetry from a Strands agent using the
Strands-only instrumentation package. It runs a real Strands agent against AWS
Bedrock and uses a simple HTTP tool, so the trace shows the agent invocation,
LLM call, and tool call.

## Requirements

- Python >= 3.10
- `strands-agents >= 1.0.0`
- OpenTelemetry SDK and OTLP exporter packages
- AWS credentials with Bedrock access
- Model access enabled for `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- A running OpenTelemetry collector, defaulting to `http://127.0.0.1:4317`

## Verify AWS Credentials

Before running, confirm your credentials are valid and have Bedrock access:

```bash
aws sts get-caller-identity

aws bedrock get-inference-profile \
  --inference-profile-identifier us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --region us-west-2
```

## Installation

```bash
pip install -e "../../[instruments]"
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## Usage

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317

# Optional overrides
export BEDROCK_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
export STRANDS_PROMPT="Fetch http://httpbin.org/json and summarize it."

python main.py
```

## What Gets Captured

The example produces this GenAI span shape:

```text
AgentInvocation
├── LLMInvocation
└── ToolCall (fetch_page)
```

It also exports GenAI metrics and optional content events through the configured
emitters.

## Configuration

By default, the instrumentation suppresses Strands' built-in OpenTelemetry tracer
to avoid double-tracing:

```bash
export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=true
```

To keep both Strands' native spans and this package's GenAI spans:

```bash
export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=false
```

Common GenAI telemetry options:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE=SPAN_AND_EVENT
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
```

## See Also

- [Strands Agents SDK](https://github.com/aws-samples/strands-agents)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk GenAI Utilities](https://github.com/signalfx/splunk-otel-python-contrib)
