# Splunk OTel Collector — Azure Container Instance

This directory contains the configuration and deployment script for running the Splunk Distribution of the OpenTelemetry Collector as an **Azure Container Instance (ACI)** gateway.

The collector receives traces and metrics from the App Service web app over OTLP gRPC (port 4317) and forwards them to Splunk Observability Cloud.

## Prerequisites

- Azure CLI installed and logged in (`az login`)
- Contributor rights on the target resource group
- Splunk Observability Cloud ingest token and realm

## Required environment variables

| Variable | Description | Example |
|---|---|---|
| `SPLUNK_ACCESS_TOKEN` | Splunk Observability Cloud ingest token | `gXgmP9v-...` |
| `SPLUNK_HEC_TOKEN` | Splunk HEC token for log ingestion | `bdef2e63-...` |
| `SPLUNK_HEC_URL` | Splunk HEC endpoint URL | `https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event` |
| `SPLUNK_REALM` | Splunk Observability Cloud realm | `us1` |
| `RESOURCE_GROUP` | Azure resource group | `my-resource-group` |
| `STORAGE_ACCOUNT` | Storage account name (globally unique, lowercase, max 24 chars) | `myotelcfgstorage` |

## Optional overrides

| Variable | Default | Description |
|---|---|---|
| `LOCATION` | `westus` | Azure region (must match the resource group) |
| `CONTAINER_NAME` | `splunk-otel-collector` | ACI container name |
| `DEPLOYMENT_ENV` | `azure` | Value for `deployment.environment` resource attribute |
| `SPLUNK_MEMORY_LIMIT_MIB` | `900` | Memory ceiling for the collector's `memory_limiter` processor |

## Deploy

```bash
export SPLUNK_ACCESS_TOKEN=<your-ingest-token>
export SPLUNK_HEC_TOKEN=<your-hec-token>
export SPLUNK_HEC_URL=https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event
export SPLUNK_REALM=us1
export RESOURCE_GROUP=<your-resource-group>
export STORAGE_ACCOUNT=<your-storage-account>

chmod +x collector/deploy-aci.sh
./collector/deploy-aci.sh
```

The script prints the container's public IP at the end. Use it to configure `OTEL_EXPORTER_OTLP_ENDPOINT` in the App Service settings:

```bash
az webapp config appsettings set \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${APP_NAME}" \
  --settings OTEL_EXPORTER_OTLP_ENDPOINT="http://<COLLECTOR_IP>:4317"
```

## Operations

```bash
# Health check
curl http://<COLLECTOR_IP>:13133/

# Tail live logs
az container logs \
  --resource-group "${RESOURCE_GROUP}" \
  --name splunk-otel-collector \
  --follow

# Restart
az container restart \
  --resource-group "${RESOURCE_GROUP}" \
  --name splunk-otel-collector

# Delete
az container delete \
  --resource-group "${RESOURCE_GROUP}" \
  --name splunk-otel-collector \
  --yes
```

## Architecture

```
App Service (Gunicorn + Uvicorn)
        │ OTLP gRPC :4317
        ▼
Azure Container Instance
  splunk-otel-collector
        │ sapm
        ├──────────────► Splunk APM (traces)
        │ signalfx
        ├──────────────► Splunk IMM (metrics)
        │ splunk_hec
        └──────────────► Splunk Log Observer (logs)
```
