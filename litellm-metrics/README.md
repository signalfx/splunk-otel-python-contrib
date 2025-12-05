# LiteLLM OpenTelemetry Metrics Debugging

Summary of steps followed for debugging litellm metrics using `hostIP`

## Chart.yaml: appVersion

| Setting | Original Value | Modified Value |
|---------|----------------|----------------|
| `appVersion` | `v1.80.5-stable` | `v1.77.7-stable` |

The appVersion was downgraded from `v1.80.5-stable` to `v1.77.7-stable` for this debugging session. `v1.80.5-stable` might have issues with reporting metrics to otel collector.

## values.yaml

### Database Configuration

For quick debugging sessions, PostgreSQL was disabled:

```yaml
postgresql:
  enabled: false
```

### Model List


```yaml
proxy_config:
  model_list:
    - model_name: gpt-3.5-turbo
      litellm_params:
        model: gpt-3.5-turbo
        api_key: your-api-key-here
  general_settings:
    master_key: os.environ/PROXY_MASTER_KEY
```

### How to Configure Host IP for OpenTelemetry Collector

For fetching the node's (collector node) IP address dynamically using hostIP add the following to `extraEnvVars`

```yaml
extraEnvVars:
  - name: SPLUNK_OTEL_AGENT
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://$(SPLUNK_OTEL_AGENT):4317"
```

**Note:** Port `4317` is the default gRPC port for OTLP. Use `4318` if your collector expects HTTP/protobuf.

---

### LiteLLM OTel Environment Variables

These environment variables control LiteLLM's OpenTelemetry integration:

| Environment Variable | Description | Example Value |
|---------------------|-------------|---------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | The endpoint of the OTel collector | `http://$(SPLUNK_OTEL_AGENT):4317` |
| `OTEL_SERVICE_NAME` | Service name that appears in traces/metrics | `lite-llm-extra-env` |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional resource attributes | `deployment.environment=lite-llm-test` |
| `LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS` | Enable OTel events | `true` |
| `LITELLM_OTEL_INTEGRATION_ENABLE_METRICS` | Enable OTel metrics | `true` |

Full `extraEnvVars` configuration:

```yaml
extraEnvVars:
  - name: SPLUNK_OTEL_AGENT
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://$(SPLUNK_OTEL_AGENT):4317"
  - name: OTEL_SERVICE_NAME
    value: "lite-llm-extra-env"
  - name: OTEL_RESOURCE_ATTRIBUTES
    value: "deployment.environment=lite-llm-test"
  - name: LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS
    value: "true"
  - name: LITELLM_OTEL_INTEGRATION_ENABLE_METRICS
    value: "true"
```

---

### LiteLLM Settings for OTel Callbacks

In the `proxy_config` section of `values.yaml`, add `litellm_settings` to enable OTel callbacks:

```yaml
proxy_config:
  litellm_settings:
    callbacks: ["otel"]
    success_callback: ["otel"]
    failure_callback: ["otel"]
    drop_params: true
```

| Setting | Description |
|---------|-------------|
| `callbacks` | List of callbacks to enable for all LLM calls |
| `success_callback` | Callbacks triggered on successful LLM calls |
| `failure_callback` | Callbacks triggered on failed LLM calls |
| `drop_params` | Drop unsupported parameters instead of raising errors |

---

## Complete OTel-Only Configuration Snippet

Add this to your existing `values.yaml` to enable OTel metrics:

```yaml
# OTel callbacks in proxy_config
proxy_config:
  # ... your existing model_list and general_settings ...
  litellm_settings:
    callbacks: ["otel"]
    success_callback: ["otel"]
    failure_callback: ["otel"]
    drop_params: true

# OTel environment variables
extraEnvVars:
  - name: SPLUNK_OTEL_AGENT
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://$(SPLUNK_OTEL_AGENT):4317"
  - name: OTEL_SERVICE_NAME
    value: "litellm-service"
  - name: OTEL_RESOURCE_ATTRIBUTES
    value: "deployment.environment=dev"
  - name: LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS
    value: "true"
  - name: LITELLM_OTEL_INTEGRATION_ENABLE_METRICS
    value: "true"
```

---

## Deploying the Chart

```bash
helm upgrade --install litellm ./litellm-helm -f values.yaml -n your-namespace
```

---

## Verifying OTel Data

1. Check that the LiteLLM pod is running:
   ```bash
   kubectl get pods -l app.kubernetes.io/name=litellm
   ```

2. Check the pod's environment variables:
   ```bash
   kubectl exec -it <pod-name> -- env | grep -E "(OTEL|LITELLM)"
   ```

3. Check for metric `gen_ai.client.operation.duration` (if sending histogram is enabled in Splunk OTel collector) or `gen_ai.client.operation.duration_count` otherwise.

---

## Updated Helm Chart Files

- [Chart.yaml](../litellm-helm/Chart.yaml)
- [values.yaml](../litellm-helm/values.yaml)
