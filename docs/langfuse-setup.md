# Langfuse Integration with Splunk Distro for OpenTelemetry

This guide explains how to forward GenAI traces to [Langfuse](https://langfuse.com/) while continuing to send telemetry to Splunk Observability Cloud using the Splunk OpenTelemetry Collector.

## Overview

Langfuse is an open-source LLM engineering platform that provides tracing, prompt management, and evaluation capabilities for LLM applications. By forwarding OpenTelemetry traces to Langfuse, you can:

- Monitor LLM application performance with detailed traces
- Manage and version prompts
- Evaluate response quality with built-in and custom metrics
- Track sessions and user interactions
- Debug issues with comprehensive trace analysis

This setup allows you to send traces to **both** Splunk and Langfuse simultaneously using the OpenTelemetry Collector's fan-out capability.

## Prerequisites

1. **Langfuse Account**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com/) or [self-host Langfuse](https://langfuse.com/self-hosting)
2. **Langfuse API Keys**: Obtain public and secret keys from your Langfuse project settings
3. **Splunk OTel Collector**: Version 0.100.0 or later recommended

### Environment Variables

Set the following environment variables for the collector. Create a credentials file at `~/.cr/.cr.langfuse`:

```bash
# ~/.cr/.cr.langfuse
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"  # or https://cloud.langfuse.com for EU

# Calculate base64 auth string: base64(public_key:secret_key)
export LANGFUSE_AUTH_STRING=$(echo -n "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" | base64)
```

Source the file before starting the collector:

```bash
source ~/.cr/.cr.langfuse
```

> **Note**: For GNU systems with long API keys, you may need to add `-w 0` to the base64 command to prevent auto-wrapping: `base64 -w 0`

## OpenTelemetry Collector Configuration

### Exporter Configuration

Add an OTLP/HTTP exporter for Langfuse in your collector configuration:

```yaml
exporters:
  # Existing Splunk exporters...
  sapm:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    endpoint: "https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"

  # Langfuse OTLP exporter
  otlphttp/langfuse:
    endpoint: "${LANGFUSE_BASE_URL}/api/public/otel"
    headers:
      Authorization: "Basic ${LANGFUSE_AUTH_STRING}"
```

> **Important**: Langfuse uses [Basic Auth](https://en.wikipedia.org/wiki/Basic_access_authentication) with base64-encoded `public_key:secret_key`. Langfuse does **not** support gRPC - use HTTP/protobuf only.

### Processor Configuration (Optional)

Add processors to inject Langfuse-specific attributes:

```yaml
processors:
  # Add Langfuse session ID from session.id attribute
  attributes/langfuse:
    actions:
      - key: langfuse.session.id
        from_attribute: session.id
        action: upsert
      - key: langfuse.user.id
        from_attribute: user.id
        action: upsert
```

### Pipeline Configuration

Configure the traces pipeline to include both Splunk and Langfuse exporters:

```yaml
service:
  pipelines:
    traces:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - attributes/langfuse      # Optional: map session/user attributes
        - resource/add_environment
      exporters:
        - sapm                     # Splunk APM
        - signalfx                 # Splunk Infrastructure Monitoring
        - otlphttp/langfuse        # Langfuse
```

### Complete Example Configuration

Here's a complete collector configuration file for dual forwarding to Splunk and Langfuse:

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

  attributes/langfuse:
    actions:
      - key: langfuse.session.id
        from_attribute: session.id
        action: upsert
      - key: langfuse.user.id
        from_attribute: user.id
        action: upsert

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

  otlphttp/langfuse:
    endpoint: "${LANGFUSE_BASE_URL}/api/public/otel"
    headers:
      Authorization: "Basic ${LANGFUSE_AUTH_STRING}"

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
        - attributes/langfuse
        - resource/add_environment
      exporters:
        - sapm
        - signalfx
        - otlphttp/langfuse
```

## Application Configuration

### Setting Session and User IDs

Langfuse uses session IDs to group related interactions and user IDs to track users. Set these attributes in your application:

```python
from opentelemetry import trace

# Set session and user ID on the current span
span = trace.get_current_span()
span.set_attribute("session.id", "chat-session-123")
span.set_attribute("user.id", "user-456")
```

The collector will automatically map these to Langfuse's expected attributes (`langfuse.session.id` and `langfuse.user.id`).

### Langfuse-Specific Attributes

You can also set Langfuse-specific attributes directly in your application for more control:

```python
from opentelemetry import trace

span = trace.get_current_span()

# Trace-level attributes
span.set_attribute("langfuse.session.id", "chat-session-123")
span.set_attribute("langfuse.user.id", "user-456")
span.set_attribute("langfuse.trace.name", "Customer Support Chat")
span.set_attribute("langfuse.trace.tags", ["production", "support"])
span.set_attribute("langfuse.release", "v1.2.3")

# Observation-level metadata (filterable in Langfuse UI)
span.set_attribute("langfuse.observation.metadata.customer_tier", "premium")
span.set_attribute("langfuse.observation.metadata.region", "us-west")
```

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

## Langfuse Attribute Mapping

Langfuse maps OpenTelemetry span attributes to its data model. Here are the key mappings:

### Trace-Level Attributes

| Langfuse Field | OpenTelemetry Attribute(s) |
|----------------|---------------------------|
| `name` | `langfuse.trace.name`, root span name |
| `userId` | `langfuse.user.id`, `user.id` |
| `sessionId` | `langfuse.session.id`, `session.id` |
| `release` | `langfuse.release` |
| `version` | `langfuse.version` |
| `tags` | `langfuse.trace.tags` |
| `metadata` | `langfuse.trace.metadata.*` |
| `environment` | `deployment.environment`, `deployment.environment.name` |

### Observation-Level Attributes

| Langfuse Field | OpenTelemetry Attribute(s) |
|----------------|---------------------------|
| `model` | `gen_ai.request.model`, `gen_ai.response.model` |
| `input` | `langfuse.observation.input`, `gen_ai.prompt` |
| `output` | `langfuse.observation.output`, `gen_ai.completion` |
| `usage` | `gen_ai.usage.*`, `llm.token_count.*` |
| `metadata` | `langfuse.observation.metadata.*` |

## Filtering Spans (Optional)

If you want to send only GenAI-related spans to Langfuse (while sending all spans to Splunk), use the filter processor:

```yaml
processors:
  filter/genai-only:
    error_mode: ignore
    traces:
      span:
        - 'attributes["gen_ai.system"] == nil'  # Drop non-GenAI spans

service:
  pipelines:
    traces/langfuse:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - filter/genai-only
        - attributes/langfuse
      exporters:
        - otlphttp/langfuse
    
    traces/splunk:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - resource/add_environment
      exporters:
        - sapm
        - signalfx
```

## Troubleshooting

### 401 Unauthorized Error

**Symptom**: Collector logs show `401 Unauthorized` errors.

**Solution**: Verify your auth string is correctly base64 encoded:
```bash
# Correct format: public_key:secret_key
echo -n "pk-lf-xxx:sk-lf-xxx" | base64
```

Ensure the Authorization header uses "Basic" prefix:
```yaml
headers:
  Authorization: "Basic ${LANGFUSE_AUTH_STRING}"
```

### 415 Unsupported Media Type

**Symptom**: Collector logs show `415 Unsupported Media Type`.

**Solution**: Langfuse only supports HTTP/protobuf (not gRPC). Ensure you're using `otlphttp` exporter (not `otlp`):
```yaml
exporters:
  otlphttp/langfuse:  # ✓ Correct
    endpoint: "..."
  
  # otlp/langfuse:    # ✗ Wrong - Langfuse doesn't support gRPC
```

### Traces Not Appearing in Langfuse

**Symptom**: No traces visible in Langfuse UI.

**Possible causes**:
1. **Endpoint URL**: Make sure you're using the correct endpoint path (`/api/public/otel`)
2. **Region**: Use `https://us.cloud.langfuse.com` for US or `https://cloud.langfuse.com` for EU
3. **Root span**: Langfuse requires a root span to create a trace correctly

### Session IDs Not Grouping Correctly

**Symptom**: Sessions are not grouped correctly in Langfuse UI.

**Solution**: 
1. Ensure `session.id` or `langfuse.session.id` is set on spans
2. Propagate the session ID to all spans in a trace using OpenTelemetry Baggage:

```python
from opentelemetry import baggage
from opentelemetry.context import attach

# Set session ID in baggage for propagation
ctx = baggage.set_baggage("session.id", "chat-session-123")
attach(ctx)
```

## Data Flow Comparison: Splunk vs Langfuse

| Feature | Splunk | Langfuse |
|---------|--------|----------|
| **Transport** | SAPM (proprietary), OTLP | OTLP HTTP/protobuf only |
| **Session ID** | `session.id` | `langfuse.session.id`, `session.id` |
| **User ID** | Custom attribute | `langfuse.user.id`, `user.id` |
| **Environment** | Resource attribute | `deployment.environment` |
| **Metadata** | Span attributes | `langfuse.*.metadata.*` (filterable) |

## Architecture Diagram

```
┌─────────────────┐     OTLP/gRPC     ┌──────────────────────────┐
│   Application   │ ─────────────────▶│   Splunk OTel Collector  │
│  (Instrumented) │                   │                          │
└─────────────────┘                   │  ┌────────────────────┐  │
                                      │  │ attributes/langfuse│  │
                                      │  └────────────────────┘  │
                                      │            │             │
                                      │     ┌──────┴──────┐      │
                                      │     ▼             ▼      │
                                      │  ┌─────┐    ┌─────────┐  │
                                      │  │SAPM │    │OTLP/HTTP│  │
                                      │  └──┬──┘    └────┬────┘  │
                                      └─────│────────────│───────┘
                                            │            │
                                            ▼            ▼
                                      ┌──────────┐  ┌─────────┐
                                      │ Splunk   │  │Langfuse │
                                      │   APM    │  │         │
                                      └──────────┘  └─────────┘
```

## References

- [Langfuse OpenTelemetry Documentation](https://langfuse.com/docs/integrations/opentelemetry)
- [Langfuse Sessions Documentation](https://langfuse.com/docs/tracing-features/sessions)
- [Langfuse Attribute Mapping](https://langfuse.com/integrations/native/opentelemetry#property-mapping)
- [Splunk OpenTelemetry Collector](https://docs.splunk.com/observability/en/gdi/opentelemetry/opentelemetry.html)
- [OpenTelemetry Collector Configuration](https://opentelemetry.io/docs/collector/configuration/)

## See Also

- [Galileo Setup Guide](galileo-setup.md) - Forward traces to Galileo alongside Splunk
- [Traceloop Setup Guide](traceloop-setup.md) - Forward traces to Traceloop alongside Splunk
