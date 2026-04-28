# Bedrock AgentCore Instrumentation Demo

This example demonstrates that OpenTelemetry instrumentation is working for all AWS Bedrock AgentCore components.

The demo shows:
- Automatic span creation for all AgentCore operations
- Proper span hierarchy (Workflow → ToolCall)
- Telemetry with correct attributes (gen_ai.system, operation names, etc.)
- Graceful handling of AWS permission errors

## What Gets Instrumented

This demo shows telemetry for:

- **BedrockAgentCoreApp.entrypoint** → `Workflow` spans
- **MemoryClient.retrieve_memories** → `RetrievalInvocation` spans
- **MemoryClient operations** (create_event, create_blob_event, list_events) → `ToolCall` spans
- **CodeInterpreter operations** (start, stop, execute_code, install_packages, upload_file) → `ToolCall` spans
- **BrowserClient operations** (start, stop, take_control, release_control, get_session) → `ToolCall` spans

## Requirements

```bash
# Install all required dependencies
pip install -r requirements.txt

# Install the instrumentation package (from repo root)
pip install -e instrumentation-genai/opentelemetry-instrumentation-bedrock-agentcore[instruments]
```

**Required packages:**
- `bedrock-agentcore>=1.0.0` - AWS Bedrock AgentCore SDK
- OpenTelemetry SDK and exporters
- Splunk GenAI utilities
- boto3 (for AWS credential checks)

## AWS Setup

1. Configure AWS credentials:
```bash
aws configure
```

2. Set your default region:
```bash
export AWS_DEFAULT_REGION=us-west-2
```

## Running the Demo

### 1. Start OpenTelemetry Collector

Using Docker:
```bash
docker run -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector:latest
```

### 2. Run the Demo

Basic run:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
python main.py
```

With content capture (captures detailed operation data):
```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
python main.py
```

## What the Demo Does

The demo creates a simple workflow that:

1. **Creates a Workflow span** using `BedrockAgentCoreApp.entrypoint`
2. **Calls AgentCore operations** that create ToolCall spans:
   - CodeInterpreter.start()
   - BrowserClient.start()
   - MemoryClient.list_memories()
3. **Shows span output** in the console so you can verify instrumentation

The demo intentionally uses operations that may fail due to AWS permissions - this is expected and demonstrates that spans are created even when operations fail.

## Viewing Telemetry

After running the demo, view the traces in:

- **Splunk Observability Cloud**: Look for service `bedrock-agentcore-demo`
- **Jaeger**: http://localhost:16686 (if using Jaeger backend)
- **Zipkin**: http://localhost:9411 (if using Zipkin backend)

### Expected Span Hierarchy

```
Workflow: research_workflow
├── RetrievalInvocation: retrieve_memories
├── ToolCall: code_interpreter.start
├── ToolCall: code_interpreter.execute_code
├── ToolCall: code_interpreter.stop
├── ToolCall: browser.start
├── ToolCall: browser.take_control
├── ToolCall: browser.release_control
├── ToolCall: browser.stop
└── ToolCall: memory.create_event
```

## Components Used

This demo uses **real** `bedrock-agentcore` components:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory.client import MemoryClient
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from bedrock_agentcore.tools.browser_client import BrowserClient
```

All operations are instrumented automatically via the `BedrockAgentCoreInstrumentor` to produce OpenTelemetry spans, metrics, and events.

## Environment Variables

- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTLP collector endpoint (default: `http://127.0.0.1:4317`)
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` - Capture detailed operation content (default: `false`)
- `AWS_DEFAULT_REGION` - AWS region (default: `us-west-2`)
- `ENVIRONMENT` - Deployment environment (default: `development`)

## Troubleshooting

### AWS Credentials Not Found
```bash
# Configure credentials
aws configure

# Or use environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

### OTLP Connection Refused
- Ensure the OpenTelemetry collector is running on port 4317
- Check firewall settings
- Verify `OTEL_EXPORTER_OTLP_ENDPOINT` is set correctly

### Import Errors
```bash
# Reinstall the instrumentation package
pip install -e instrumentation-genai/opentelemetry-instrumentation-bedrock-agentcore[instruments]

# Ensure OpenTelemetry packages are installed
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## Additional Resources

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Splunk Observability Cloud](https://www.splunk.com/en_us/products/observability.html)
