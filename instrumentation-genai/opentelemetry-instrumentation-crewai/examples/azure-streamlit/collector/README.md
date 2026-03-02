# Splunk OTel Collector on Azure Container Instance (Gateway Mode)

This guide deploys the [Splunk distribution of the OpenTelemetry Collector](https://docs.splunk.com/observability/en/gdi/opentelemetry/opentelemetry.html) as a **gateway** in **Azure Container Instance (ACI)**. The gateway receives OTLP telemetry from the Streamlit + CrewAI App Service and forwards traces, metrics, and logs to **Splunk Observability Cloud**.

## Architecture

```
Azure App Service (Streamlit + CrewAI)
  │
  │  OTLP/gRPC  port 4317
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│          Azure Container Instance                           │
│          splunk-otel-collector (gateway mode)               │
│                                                             │
│  Receivers:  otlp (4317/4318)  jaeger (14250/14268/6831)    │
│              zipkin (9411)     signalfx (9943)              │
│  Processors: memory_limiter → batch → resourcedetection     │
│              resource/add_environment                       │
│  Exporters:  sapm → Splunk APM (traces)                     │
│              signalfx → Splunk Infrastructure (metrics)     │
│              splunk_hec → Splunk Log Observer (logs)        │
│                                                             │
│  Config mounted from:  Azure File Share (otelconfig)        │
└─────────────────────────────────────────────────────────────┘
  │
  │  SAPM / SignalFx / HEC
  ▼
Splunk Observability Cloud
```

**Why gateway mode?**
In gateway mode the collector runs as a standalone service that any number of applications can send telemetry to. A single collector handles batching, retry, and credential management so individual apps stay credential-free.

---

## Files

| File | Purpose |
|---|---|
| `otel-collector-config.yaml` | Full collector pipeline configuration |
| `deploy-aci.sh` | One-shot deployment: creates storage, uploads config, creates ACI |

---

## Prerequisites

- Azure CLI ≥ 2.50 installed and authenticated (`az login`)
- Contributor rights on the target resource group
- `Microsoft.ContainerInstance` resource provider registered:

```bash
az provider register --namespace Microsoft.ContainerInstance --wait
az provider show --namespace Microsoft.ContainerInstance --query registrationState
```

- A Splunk Observability Cloud access token with **ingest** permissions
- A Splunk HEC (HTTP Event Collector) token for log ingestion

---

## Quick deploy

```bash
export SPLUNK_ACCESS_TOKEN=<your-token>
export SPLUNK_HEC_TOKEN=<your-hec-token>
export SPLUNK_HEC_URL=https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event
export SPLUNK_REALM=us1
export RESOURCE_GROUP=<your-resource-group>
export STORAGE_ACCOUNT=<your-storage-account>

cd collector
chmod +x deploy-aci.sh
./deploy-aci.sh
```

The script performs four steps automatically:

1. Creates an Azure Storage Account and File Share
2. Uploads `otel-collector-config.yaml` to the file share
3. Creates the ACI container with the config file mounted
4. Prints the public IP and the `OTEL_EXPORTER_OTLP_ENDPOINT` value to set in the app

---

## Manual step-by-step deployment

### Step 1 — Create storage for the config file

ACI cannot load config files from the local filesystem. A config file must be hosted on an Azure File Share that is mounted into the container at runtime.

```bash
RESOURCE_GROUP="o11y-inframon-ai"
STORAGE_ACCOUNT="otelcfgo11yai"   # globally unique, lowercase, max 24 chars
FILE_SHARE="otelconfig"
LOCATION="westus"

az storage account create \
  --name "${STORAGE_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --sku Standard_LRS \
  --kind StorageV2

STORAGE_KEY=$(az storage account keys list \
  --resource-group "${RESOURCE_GROUP}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --query "[0].value" --output tsv)

az storage share create \
  --name "${FILE_SHARE}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --account-key "${STORAGE_KEY}"
```

### Step 2 — Upload the collector config

```bash
az storage file upload \
  --share-name "${FILE_SHARE}" \
  --source otel-collector-config.yaml \
  --path otel-collector-config.yaml \
  --account-name "${STORAGE_ACCOUNT}" \
  --account-key "${STORAGE_KEY}"
```

### Step 3 — Create the container instance

```bash
SPLUNK_REALM="us1"

az container create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "splunk-otel-collector" \
  --image "quay.io/signalfx/splunk-otel-collector:latest" \
  --os-type Linux \
  --cpu 1 \
  --memory 2 \
  --restart-policy Always \
  --ports 4317 4318 13133 6060 9411 \
  --ip-address Public \
  --azure-file-volume-account-name "${STORAGE_ACCOUNT}" \
  --azure-file-volume-account-key "${STORAGE_KEY}" \
  --azure-file-volume-share-name "${FILE_SHARE}" \
  --azure-file-volume-mount-path /etc/otel \
  --command-line "/otelcol --config /etc/otel/otel-collector-config.yaml" \
  --environment-variables \
    SPLUNK_ACCESS_TOKEN="<your-access-token>" \
    SPLUNK_HEC_TOKEN="<your-hec-token>" \
    SPLUNK_HEC_URL="https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event" \
    SPLUNK_API_URL="https://api.${SPLUNK_REALM}.signalfx.com" \
    SPLUNK_INGEST_URL="https://ingest.${SPLUNK_REALM}.signalfx.com" \
    SPLUNK_TRACE_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace" \
    SPLUNK_REALM="${SPLUNK_REALM}" \
    SPLUNK_BUNDLE_DIR="/usr/lib/splunk-otel-collector/agent-bundle" \
    SPLUNK_COLLECTD_DIR="/usr/lib/splunk-otel-collector/agent-bundle/run/collectd" \
    SPLUNK_LISTEN_INTERFACE="0.0.0.0" \
    SPLUNK_MEMORY_LIMIT_MIB="900" \
    OTEL_RESOURCE_ATTRIBUTES="deployment.environment=<your-env>" \
  --output table
```

> **ACI port limit:** Azure Container Instance allows a maximum of **5 public IP ports**. The configuration exposes: `4317` (OTLP gRPC), `4318` (OTLP HTTP), `13133` (health check), `6060` (http_forwarder), `9411` (Zipkin). Additional receiver ports (Jaeger, SignalFx) are available internally but not exposed publicly.

### Step 4 — Get the public IP

```bash
COLLECTOR_IP=$(az container show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "splunk-otel-collector" \
  --query "ipAddress.ip" --output tsv)

echo "OTEL_EXPORTER_OTLP_ENDPOINT=http://${COLLECTOR_IP}:4317"
```

Set this value as the `OTEL_EXPORTER_OTLP_ENDPOINT` application setting on the App Service.

---

## Environment variables reference

| Variable | Description | Example |
|---|---|---|
| `SPLUNK_ACCESS_TOKEN` | Ingest token for Splunk O11y Cloud | `gXgmP9v-...` |
| `SPLUNK_HEC_TOKEN` | HEC token for log ingestion | `bdef2e63-...` |
| `SPLUNK_HEC_URL` | Splunk HEC endpoint | `https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event` |
| `SPLUNK_REALM` | Splunk O11y Cloud realm | `us1` |
| `SPLUNK_API_URL` | SignalFx API URL (derived from realm) | `https://api.us1.signalfx.com` |
| `SPLUNK_INGEST_URL` | Splunk ingest URL (derived from realm) | `https://ingest.us1.signalfx.com` |
| `SPLUNK_TRACE_URL` | SAPM trace ingest URL (derived from realm) | `https://ingest.us1.signalfx.com/v2/trace` |
| `SPLUNK_LISTEN_INTERFACE` | Interface for all receivers | `0.0.0.0` |
| `SPLUNK_MEMORY_LIMIT_MIB` | Memory ceiling for the memory_limiter processor | `900` |
| `SPLUNK_BUNDLE_DIR` | Path to the agent bundle inside the container | `/usr/lib/splunk-otel-collector/agent-bundle` |
| `SPLUNK_COLLECTD_DIR` | Path to collectd inside the container | `/usr/lib/splunk-otel-collector/agent-bundle/run/collectd` |
| `OTEL_RESOURCE_ATTRIBUTES` | Resource attributes added to all telemetry | `deployment.environment=production` |

---

## Operations

### Health check

```bash
COLLECTOR_IP=$(az container show \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector \
  --query "ipAddress.ip" --output tsv)

curl http://${COLLECTOR_IP}:13133/
# Expected: {"status": "Server available","upSince":"...","uptime":"..."}
```

### View live logs

```bash
az container logs \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector \
  --follow
```

### Check container state

```bash
az container show \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector \
  --query "{state:instanceView.state,ip:ipAddress.ip,restartCount:containers[0].instanceView.restartCount}" \
  --output table
```

### Stop the container (pause billing)

```bash
az container stop \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector
```

### Start the container

```bash
az container start \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector
```

### Restart the container

```bash
az container restart \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector
```

### Update the config file and restart

```bash
# Re-upload modified config
az storage file upload \
  --share-name otelconfig \
  --source otel-collector-config.yaml \
  --path otel-collector-config.yaml \
  --account-name otelcfgo11yai \
  --account-key "${STORAGE_KEY}" \
  --overwrite

# Restart to pick up changes
az container restart \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector
```

### Delete the deployment

```bash
az container delete \
  --resource-group o11y-inframon-ai \
  --name splunk-otel-collector \
  --yes
```

---

## Pipeline overview

The collector config defines the following pipelines:

| Pipeline | Receivers | Processors | Exporters |
|---|---|---|---|
| `traces` | `otlp`, `jaeger`, `zipkin` | `memory_limiter` → `batch` → `resourcedetection` → `resource/add_environment` | `sapm`, `signalfx` |
| `metrics` | `otlp`, `signalfx` | `memory_limiter` → `batch` → `resourcedetection` | `signalfx` |
| `metrics/internal` | `prometheus/internal` | `memory_limiter` → `batch` → `resourcedetection` | `signalfx` |
| `logs` | `otlp` | `memory_limiter` → `batch` → `resourcedetection` → `resource/add_environment` | `splunk_hec`, `splunk_hec/profiling` |
| `logs/signalfx` | `signalfx` | `memory_limiter` → `batch` → `resourcedetection` | `signalfx` |

The `resourcedetection` processor automatically attaches Azure resource metadata (subscription, resource group, VM/container name) to all telemetry.

---

## Troubleshooting

### Container keeps restarting

Check the logs for config parse errors:
```bash
az container logs --resource-group o11y-inframon-ai --name splunk-otel-collector
```

Common causes:
- Deprecated YAML keys. The `service.telemetry.metrics.address` field was removed in newer collector versions; use `level: basic` instead (already correct in this config).
- Missing required environment variables (`SPLUNK_ACCESS_TOKEN`, etc.).

### No traces appearing in Splunk APM

1. Verify the collector is healthy: `curl http://<ip>:13133/`
2. Check the `OTEL_EXPORTER_OTLP_ENDPOINT` in the App Service matches the collector IP.
3. Confirm `SPLUNK_ACCESS_TOKEN` has **ingest** scope.
4. Check App Service logs for `[OTel]` lines confirming export attempts.

### ACI registration error

```
(MissingSubscriptionRegistration) The subscription is not registered to use namespace 'Microsoft.ContainerInstance'
```

Fix:
```bash
az provider register --namespace Microsoft.ContainerInstance --wait
```

---

## References

- [Splunk OTel Collector documentation](https://docs.splunk.com/observability/en/gdi/opentelemetry/opentelemetry.html)
- [Azure Container Instances documentation](https://learn.microsoft.com/en-us/azure/container-instances/)
- [Splunk OTel Collector image on Quay.io](https://quay.io/repository/signalfx/splunk-otel-collector)
- [OpenTelemetry Collector configuration](https://opentelemetry.io/docs/collector/configuration/)

## License

Apache-2.0
