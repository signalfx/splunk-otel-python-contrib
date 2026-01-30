# Galileo Integration with Splunk Distro for OpenTelemetry

This guide explains how to forward GenAI traces to [Galileo](https://www.rungalileo.io/) while continuing to send telemetry to Splunk Observability Cloud using the Splunk OpenTelemetry Collector.

## Overview

Galileo is an AI observability platform that provides evaluation, monitoring, and debugging capabilities for LLM applications. By forwarding OpenTelemetry traces to Galileo, you can:

- Monitor LLM application performance in real-time
- Evaluate response quality with built-in metrics
- Debug issues with detailed trace analysis
- Track sessions across multiple interactions

This setup allows you to send traces to **both** Splunk and Galileo simultaneously using the OpenTelemetry Collector's fan-out capability.

## Prerequisites

1. **Galileo Account**: Sign up at [rungalileo.io](https://www.rungalileo.io/)
2. **Galileo API Key**: Obtain from your Galileo console
3. **Galileo Project**: Create a project in Galileo for your application
4. **Splunk OTel Collector**: Version 0.100.0 or later recommended

### Environment Variables

Set the following environment variables for the collector:

```bash
export GALILEO_API_KEY="your-galileo-api-key"
export GALILEO_PROJECT="your-project-name"
export GALILEO_LOG_STREAM="default"  # or your custom log stream name
```

## OpenTelemetry Collector Configuration

### Exporter Configuration

Add an OTLP/HTTP exporter for Galileo in your collector configuration:

```yaml
exporters:
  # Existing Splunk exporters...
  sapm:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    endpoint: "https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"

  # Galileo OTLP exporter
  otlphttp/galileo:
    endpoint: "https://api.galileo.ai"
    headers:
      Galileo-API-Key: "${GALILEO_API_KEY}"
      project: "${GALILEO_PROJECT}"
      logstream: "${GALILEO_LOG_STREAM}"
    retry_on_failure:
      enabled: false  # Required: Galileo returns "{}" which causes parsing errors
```

> **Important**: The `retry_on_failure: enabled: false` setting is required because Galileo's API returns an empty JSON response (`"{}"`) that the collector's default retry logic interprets as a parsing error. This is benign - traces are successfully ingested.

### Processor Configuration

Add processors to inject Galileo-specific attributes:

```yaml
processors:
  # Add Galileo resource attributes
  resource/galileo:
    attributes:
      - key: galileo.project.name
        value: "${GALILEO_PROJECT}"
        action: upsert
      - key: galileo.logstream.name
        value: "${GALILEO_LOG_STREAM}"
        action: upsert

  # Rename session.id to galileo.session.id for Galileo compatibility
  attributes/galileo:
    actions:
      - key: galileo.session.id
        from_attribute: session.id
        action: upsert
```

### Pipeline Configuration

Configure the traces pipeline to include both Splunk and Galileo exporters:

```yaml
service:
  pipelines:
    traces:
      receivers:
        - otlp
      processors:
        - memory_limiter
        - batch
        - resource/galileo      # Add Galileo resource attributes
        - attributes/galileo    # Rename session.id for Galileo
        - resource/add_environment
      exporters:
        - sapm                  # Splunk APM
        - signalfx              # Splunk Infrastructure Monitoring
        - otlphttp/galileo      # Galileo
```

### Complete Example Configuration

Here's a complete collector configuration file for dual forwarding to Splunk and Galileo:

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

  resource/galileo:
    attributes:
      - key: galileo.project.name
        value: "${GALILEO_PROJECT}"
        action: upsert
      - key: galileo.logstream.name
        value: "${GALILEO_LOG_STREAM}"
        action: upsert

  attributes/galileo:
    actions:
      - key: galileo.session.id
        from_attribute: session.id
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

  otlphttp/galileo:
    endpoint: "https://api.galileo.ai"
    headers:
      Galileo-API-Key: "${GALILEO_API_KEY}"
      project: "${GALILEO_PROJECT}"
      logstream: "${GALILEO_LOG_STREAM}"
    retry_on_failure:
      enabled: false

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
        - resource/galileo
        - attributes/galileo
        - resource/add_environment
      exporters:
        - sapm
        - signalfx
        - otlphttp/galileo
```

## Application Configuration

### Setting Session IDs

Galileo uses session IDs to group related interactions. The instrumentation library supports setting session IDs via span attributes.

Set the `session.id` attribute in your application, and the collector will automatically rename it to `galileo.session.id`:

```python
from opentelemetry import trace

# Set session ID on the current span
span = trace.get_current_span()
span.set_attribute("session.id", "user-session-123")
```

For the SRE Incident Copilot example application, use the `--session-id` parameter:

```bash
python main.py --scenario scenario-001 --session-id my-session-id
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

## Galileo-Specific Attributes

The following attributes are recognized by Galileo:

| Attribute | Description | Set By |
|-----------|-------------|--------|
| `galileo.project.name` | Galileo project name | Collector (resource processor) |
| `galileo.logstream.name` | Galileo log stream name | Collector (resource processor) |
| `galileo.session.id` | Session identifier for grouping interactions | Application (via `session.id`) → Collector renames |

## Troubleshooting

### 401 Unauthorized Error

**Symptom**: Collector logs show `401 Unauthorized` errors.

**Solution**: Ensure you're using the correct header format. Galileo requires:
```yaml
headers:
  Galileo-API-Key: "${GALILEO_API_KEY}"  # NOT "Authorization: Bearer ..."
```

### 415 Unsupported Media Type

**Symptom**: Collector logs show `415 Unsupported Media Type`.

**Solution**: Do NOT set `encoding: json`. Galileo expects protobuf encoding (the default):
```yaml
otlphttp/galileo:
  endpoint: "https://api.galileo.ai"
  # Do NOT add: encoding: json
```

### JSON Parsing Errors (Benign)

**Symptom**: Collector logs show errors like:
```
error decoding response: invalid character '"' looking for beginning of value
```

**Explanation**: This is benign. Galileo returns `"{}"` (a JSON string containing "{}") instead of standard OTLP response format. Traces are still successfully ingested.

**Solution**: Disable retries to suppress error spam:
```yaml
otlphttp/galileo:
  retry_on_failure:
    enabled: false
```

### Session IDs Not Appearing in Galileo

**Symptom**: Sessions are not grouped correctly in Galileo UI.

**Solution**: Ensure the attributes processor is in the pipeline and ordering is correct:
1. Verify `attributes/galileo` processor is defined
2. Verify it's included in the `traces` pipeline processors list
3. Check that your application sets `session.id` span attribute

## Architecture Diagram

```
┌─────────────────┐     OTLP/gRPC     ┌──────────────────────────┐
│   Application   │ ─────────────────▶│   Splunk OTel Collector  │
│  (Instrumented) │                   │                          │
└─────────────────┘                   │  ┌────────────────────┐  │
                                      │  │ resource/galileo   │  │
                                      │  │ attributes/galileo │  │
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
                                      │ Splunk   │  │ Galileo │
                                      │   APM    │  │   AI    │
                                      └──────────┘  └─────────┘
```

## See Also

- [Langfuse Setup Guide](langfuse-setup.md) - Forward traces to Langfuse alongside Splunk
- [Traceloop Setup Guide](traceloop-setup.md) - Forward traces to Traceloop alongside Splunk
