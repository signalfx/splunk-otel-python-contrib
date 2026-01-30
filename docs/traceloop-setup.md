# Traceloop Integration with Splunk Distro for OpenTelemetry

This guide explains how to forward GenAI traces to [Traceloop](https://www.traceloop.com/) while continuing to send telemetry to Splunk Observability Cloud using the Splunk OpenTelemetry Collector.

## Overview

Traceloop is an LLM observability and evaluation platform that provides monitoring, debugging, and quality assurance for LLM applications. By forwarding OpenTelemetry traces to Traceloop, you can:

- Monitor LLM application performance in real-time
- Evaluate and test prompt changes before deployment
- Debug issues with detailed trace visualization
- Track costs and usage across different models
- Organize traces by projects and environments

This setup allows you to send traces to **both** Splunk and Traceloop simultaneously using the OpenTelemetry Collector's fan-out capability.

## Prerequisites

1. **Traceloop Account**: Sign up at [app.traceloop.com](https://app.traceloop.com/)
2. **Traceloop API Key**: Generate from [Settings → API Keys](https://app.traceloop.com/settings/api-keys)
3. **Splunk OTel Collector**: Version 0.100.0 or later recommended

### Environment Variables

Create a credentials file at `~/.cr/.cr.traceloop`:

```bash
# ~/.cr/.cr.traceloop
export TRACELOOP_API_KEY="your-traceloop-api-key-here"
export TRACELOOP_BASE_URL="https://api.traceloop.com"
```

Source the file before starting the collector:

```bash
source ~/.cr/.cr.traceloop
```

> **Note**: API keys are scoped to a specific project and environment. Make sure you're viewing the correct project and environment in the Traceloop dashboard that matches your API key.

## OpenTelemetry Collector Configuration

### Exporter Configuration

Add an OTLP/HTTP exporter for Traceloop in your collector configuration:

```yaml
exporters:
  # Existing Splunk exporters...
  sapm:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    endpoint: "https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"

  # Traceloop OTLP exporter
  otlphttp/traceloop:
    endpoint: "${TRACELOOP_BASE_URL}"
    headers:
      "Authorization": "Bearer ${TRACELOOP_API_KEY}"
```

> **Note**: Traceloop uses Bearer token authentication, which is simpler than Langfuse's Basic Auth or Galileo's custom headers.

### Pipeline Configuration

Configure the traces pipeline to include both Splunk and Traceloop exporters:

```yaml
service:
  pipelines:
    traces:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - resource/add_environment
      exporters:
        - sapm                     # Splunk APM
        - signalfx                 # Splunk Infrastructure Monitoring
        - otlphttp/traceloop       # Traceloop
```

### Complete Example Configuration

Here's a complete collector configuration file for dual forwarding to Splunk and Traceloop:

```yaml
extensions:
  health_check:
    endpoint: 0.0.0.0:13133
  zpages:
    endpoint: 0.0.0.0:55679

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  memory_limiter:
    check_interval: 2s
    limit_mib: 512

  resource/add_environment:
    attributes:
      - key: deployment.environment
        value: "${OTEL_RESOURCE_ATTRIBUTES}"
        action: upsert

exporters:
  sapm:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    endpoint: "https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"

  signalfx:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    realm: "${SPLUNK_REALM}"

  otlphttp/traceloop:
    endpoint: "${TRACELOOP_BASE_URL}"
    headers:
      "Authorization": "Bearer ${TRACELOOP_API_KEY}"

service:
  extensions:
    - health_check
    - zpages
  pipelines:
    traces:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - resource/add_environment
      exporters:
        - sapm
        - signalfx
        - otlphttp/traceloop
```

## Application Configuration

### Environment Variables for Applications

Configure your instrumented application with these environment variables:

```bash
# Standard OTEL configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_SERVICE_NAME="your-service-name"

# GenAI instrumentation settings
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE="SPAN_AND_EVENT"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event"
```

### Using OpenLLMetry SDK Directly (Alternative)

If you prefer to send traces directly to Traceloop without going through the collector, you can use the OpenLLMetry SDK:

```python
from traceloop.sdk import Traceloop

# Initialize with API key
Traceloop.init()  # Uses TRACELOOP_API_KEY env var

# Or initialize with explicit configuration
Traceloop.init(
    api_key="your-api-key",
    disable_batch=True  # For local development
)
```

However, using the OpenTelemetry Collector provides more flexibility for multi-destination routing.

## Traceloop Attribute Mapping

Traceloop uses the OpenLLMetry semantic conventions which are compatible with OpenTelemetry GenAI semantic conventions:

| OpenTelemetry Attribute | Traceloop Usage |
|------------------------|-----------------|
| `gen_ai.system` | LLM provider (openai, anthropic, etc.) |
| `gen_ai.request.model` | Model name |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.prompt` | Prompt content |
| `gen_ai.completion` | Completion content |

## Projects and Environments

Traceloop organizes traces by **projects** and **environments**:

- **Projects**: Separate different applications or use cases
- **Environments**: Separate Development, Staging, Production

API keys are scoped to a specific project and environment. When generating an API key:

1. Go to [Settings → Organization](https://app.traceloop.com/settings/api-keys)
2. Click on your project (or create a new one)
3. Select an environment (Development, Staging, Production, or custom)
4. Click Generate API key
5. Copy the key immediately - it won't be shown again

## Troubleshooting

### 401 Unauthorized Error

**Symptom**: Collector logs show `401 Unauthorized` errors.

**Solutions**:
1. Verify `TRACELOOP_API_KEY` environment variable is set correctly
2. Ensure the API key hasn't expired or been revoked
3. Check that the Authorization header format is `Bearer <key>` (not `Basic`):

```yaml
headers:
  "Authorization": "Bearer ${TRACELOOP_API_KEY}"  # ✓ Correct
  # "Authorization": "Basic ${TRACELOOP_API_KEY}"  # ✗ Wrong
```

### Traces Not Appearing in Dashboard

**Symptom**: No traces visible in Traceloop dashboard.

**Possible causes**:
1. **Wrong project/environment**: Make sure you're viewing the project and environment that matches your API key
2. **Endpoint URL**: Ensure you're using `https://api.traceloop.com` (no trailing path)
3. **Network issues**: Check if the collector can reach the Traceloop API

### Connection Timeout

**Symptom**: Collector logs show timeout errors.

**Solution**: Add timeout configuration to the exporter:

```yaml
otlphttp/traceloop:
  endpoint: "${TRACELOOP_BASE_URL}"
  headers:
    "Authorization": "Bearer ${TRACELOOP_API_KEY}"
  timeout: 30s
```

## Comparison: Traceloop vs Other Platforms

| Feature | Traceloop | Galileo | Langfuse |
|---------|-----------|---------|----------|
| **Auth Method** | Bearer token | Custom header | Basic Auth |
| **Endpoint** | `https://api.traceloop.com` | `https://api.galileo.ai/otel/traces` | `${BASE_URL}/api/public/otel` |
| **Session Support** | Via OpenLLMetry SDK | `galileo.session.id` | `langfuse.session.id` |
| **Projects** | Yes (via API key scope) | Yes (via header) | Yes (via API key) |
| **Open Source** | OpenLLMetry SDK | No | Yes |

## Architecture Diagram

```
┌─────────────────┐     OTLP/gRPC     ┌──────────────────────────┐
│   Application   │ ─────────────────▶│   Splunk OTel Collector  │
│  (Instrumented) │                   │                          │
└─────────────────┘                   │            │             │
                                      │     ┌──────┴──────┐      │
                                      │     ▼             ▼      │
                                      │  ┌─────┐    ┌─────────┐  │
                                      │  │SAPM │    │OTLP/HTTP│  │
                                      │  └──┬──┘    └────┬────┘  │
                                      └─────│────────────│───────┘
                                            │            │
                                            ▼            ▼
                                      ┌──────────┐  ┌──────────┐
                                      │ Splunk   │  │Traceloop │
                                      │   APM    │  │          │
                                      └──────────┘  └──────────┘
```

## References

- [Traceloop Documentation](https://www.traceloop.com/docs)
- [OpenLLMetry Integration with Traceloop](https://www.traceloop.com/docs/openllmetry/integrations/traceloop)
- [OpenLLMetry Python SDK](https://www.traceloop.com/docs/openllmetry/getting-started-python)
- [Traceloop Projects and Environments](https://www.traceloop.com/docs/settings/projects-and-environments)
- [Splunk OpenTelemetry Collector](https://docs.splunk.com/observability/en/gdi/opentelemetry/opentelemetry.html)

## See Also

- [Galileo Setup Guide](galileo-setup.md) - Forward traces to Galileo alongside Splunk
- [Langfuse Setup Guide](langfuse-setup.md) - Forward traces to Langfuse alongside Splunk
