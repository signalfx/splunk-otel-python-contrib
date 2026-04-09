# Strands Instrumentation Example

This example demonstrates how to use the OpenTelemetry Strands instrumentation to capture telemetry from Strands agents.

## Requirements

- Python >= 3.10
- strands-agents >= 1.0.0
- AWS credentials configured (for Bedrock model calls)

## Installation

```bash
# Install the instrumentation package
pip install -e ../../

# Install Strands SDK (if available)
pip install strands-agents

# Or install with test dependencies
pip install -e "../../[instruments,test]"
```

## Usage

```bash
# Set up AWS credentials
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Run the example
python main.py
```

## What Gets Captured

The instrumentation captures:

1. **Agent Invocations** - Spans for `Agent.__call__()` and `Agent.invoke_async()`
   - Agent name, model, instructions
   - Input/output messages
   - Tool list

2. **LLM Calls** - Spans for model invocations via Strands hooks
   - Model ID, provider
   - Input/output messages
   - Token usage
   - Temperature, max_tokens, etc.

3. **Tool Calls** - Spans for tool executions via Strands hooks
   - Tool name, description
   - Tool arguments
   - Tool results

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
# Control content capture
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

# Configure emitters
export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span_metric_event
```

## Expected Output

When you run the example, you should see:

1. Console logs showing the agent invocation
2. OpenTelemetry span exports showing:
   - `gen_ai.operation.name: agent_invocation` (Agent span)
   - `gen_ai.operation.name: llm_invocation` (LLM spans)
   - `gen_ai.operation.name: execute_tool` (Tool spans)

## Troubleshooting

### Import Error: strands-agents not found

```bash
pip install strands-agents
```

### AWS Credentials Error

Make sure AWS credentials are configured:

```bash
aws configure
# or set environment variables
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### No Spans Exported

Check that:
1. Instrumentation is called before importing Strands
2. Tracer provider is configured
3. Spans are flushed at the end

## See Also

- [Strands Agents SDK](https://github.com/aws-samples/strands-agents)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Splunk GenAI Utilities](https://github.com/signalfx/splunk-otel-python-contrib)
