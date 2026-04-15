# Strands Instrumentation Example

This example demonstrates how to use the OpenTelemetry Strands instrumentation to capture telemetry from Strands agents. It runs a real agent against Bedrock that uses two HTTP tools (`fetch_page`, `get_ip_info`) backed by public APIs — no mock data.

## Requirements

- Python >= 3.10
- strands-agents >= 1.0.0
- bedrock-agentcore >= 1.0.0
- opentelemetry-sdk + OTLP exporter packages
- AWS credentials with Bedrock access
- Model access enabled in the Bedrock console for `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- A running OpenTelemetry collector (default: `http://127.0.0.1:4317`)

## Verify AWS Credentials

Before running, confirm your credentials are valid and have Bedrock access:

```bash
# Check credentials are resolved
aws sts get-caller-identity

# Confirm the cross-region inference profile is accessible in your region
aws bedrock get-inference-profile \
  --inference-profile-identifier us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --region us-west-2
```

If `get-inference-profile` returns an `AccessDeniedException`, you need to enable model access:
1. Go to AWS Console → Bedrock → Model access
2. Request access to **Claude 3.5 Sonnet v2** under Anthropic
3. Wait for access to be granted, then re-run the check above

## Installation

```bash
# Install the instrumentation package with Strands and AgentCore dependencies
pip install -e "../../[instruments]"

# Install OpenTelemetry SDK and OTLP exporters
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## Usage

```bash
# AWS credentials
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Collector endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317

# Optional: enable Memory demo (requires an existing AgentCore memory resource)
export MEMORY_ID=your-agentcore-memory-id

# Run the example
python main.py
```

## What Gets Captured

All components run within a single `BedrockAgentCoreApp.entrypoint`, producing this span hierarchy:

```
Workflow
├── RetrievalInvocation  (memory.retrieve_memories — load prior context)  [requires MEMORY_ID]
├── AgentInvocation
│    ├── LLMInvocation
│    └── ToolCall        (fetch_page)
├── ToolCall             (memory.create_event — store result)              [requires MEMORY_ID]
├── ToolCall             (code_interpreter.start / execute / stop)         [requires AgentCore access]
└── ToolCall             (browser.start / get_session / stop)              [requires AgentCore access]
```

**Requirement 1 — Agents and tools telemetry:** AgentInvocation, LLMInvocation, ToolCall spans

**Requirement 2 — AgentCore components monitoring:** Memory retrieve as RetrievalInvocation; Memory write, CodeInterpreter, and Browser as ToolCall spans — gracefully skipped if infrastructure is not available

**Requirement 3 — Observability integration:** All telemetry exported via OTLP to collector and printed to console

## Configuration

### Suppress Built-in Tracer

By default, the instrumentation suppresses Strands' built-in OTel tracer to avoid double-tracing:

```bash
# Default behavior (suppressed)
export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=true

# Keep both Strands and instrumentation spans
export OTEL_INSTRUMENTATION_STRANDS_SUPPRESS_BUILTIN_TRACER=false
```

### GenAI Telemetry Options

```bash
# Capture message content in spans
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# Configure emitters
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
```

## Troubleshooting

### Import Error: strands-agents not found

```bash
pip install -e "../../[instruments]"
```

### AWS Credentials Error

```bash
aws configure
# or set environment variables
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Missing Dependency: botocore[crt] (SSO users only)

If you authenticate via AWS SSO and see `Missing Dependency: Using the login credential provider requires an additional dependency`:

```bash
pip install "botocore[crt]"
```

This installs the AWS Common Runtime required by the SSO credential provider. Not needed if using static IAM credentials.

### Bedrock Model Access Error

See **Verify AWS Credentials** above. The model must be explicitly enabled per AWS account.

### No Spans Exported

- Confirm your collector is running and reachable at `OTEL_EXPORTER_OTLP_ENDPOINT`
- Spans also print to console — if you see those but not in your collector, it's a network/config issue
- Check that instrumentation is called before importing Strands

## See Also

- [Strands Agents SDK](https://github.com/aws-samples/strands-agents)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk GenAI Utilities](https://github.com/signalfx/splunk-otel-python-contrib)
