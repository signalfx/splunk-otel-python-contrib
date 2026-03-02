#!/usr/bin/env bash
# =============================================================================
# deploy-aci.sh — Deploy Splunk OTel Collector to Azure Container Instance
#
# Prerequisites:
#   - Azure CLI installed and logged in  (az login)
#   - Contributor rights on the resource group
#
# Required environment variables (set before running):
#   SPLUNK_ACCESS_TOKEN   Splunk Observability Cloud ingest token
#   SPLUNK_HEC_TOKEN      Splunk HEC token for log ingestion
#   SPLUNK_HEC_URL        Splunk HEC endpoint URL
#   SPLUNK_REALM          Splunk realm (e.g. us1, eu0)
#   RESOURCE_GROUP        Azure resource group
#   STORAGE_ACCOUNT       Storage account name (globally unique, lowercase, max 24 chars)
#
# Optional overrides (defaults shown below):
#   LOCATION              Azure region                  (default: westus)
#   CONTAINER_NAME        ACI container name            (default: splunk-otel-collector)
#   DEPLOYMENT_ENV        deployment.environment tag    (default: azure)
#   SPLUNK_MEMORY_LIMIT_MIB  Memory ceiling in MiB      (default: 900)
#
# Usage:
#   export SPLUNK_ACCESS_TOKEN=<token>
#   export SPLUNK_HEC_TOKEN=<token>
#   export SPLUNK_HEC_URL=https://http-inputs-<realm>.splunkcloud.com:443/services/collector/event
#   export SPLUNK_REALM=us1
#   export RESOURCE_GROUP=<your-resource-group>
#   export STORAGE_ACCOUNT=<your-storage-account>
#   chmod +x deploy-aci.sh && ./deploy-aci.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------
: "${SPLUNK_ACCESS_TOKEN:?Required env var SPLUNK_ACCESS_TOKEN is not set}"
: "${SPLUNK_HEC_TOKEN:?Required env var SPLUNK_HEC_TOKEN is not set}"
: "${SPLUNK_HEC_URL:?Required env var SPLUNK_HEC_URL is not set}"
: "${SPLUNK_REALM:?Required env var SPLUNK_REALM is not set}"

# ---------------------------------------------------------------------------
# Config — override via env vars or edit defaults here
# ---------------------------------------------------------------------------
: "${RESOURCE_GROUP:?Required env var RESOURCE_GROUP is not set}"
LOCATION="${LOCATION:-westus}"                         # must match your RG location

: "${STORAGE_ACCOUNT:?Required env var STORAGE_ACCOUNT is not set (globally unique, lowercase, max 24 chars)}"
FILE_SHARE="${FILE_SHARE:-otelconfig}"
CONFIG_FILE="$(dirname "$0")/otel-collector-config.yaml"

CONTAINER_NAME="${CONTAINER_NAME:-splunk-otel-collector}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-quay.io/signalfx/splunk-otel-collector:latest}"

DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-azure}"

# Derived from realm
SPLUNK_API_URL="https://api.${SPLUNK_REALM}.signalfx.com"
SPLUNK_INGEST_URL="https://ingest.${SPLUNK_REALM}.signalfx.com"
SPLUNK_TRACE_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"

# Container paths for the Splunk OTel Collector image (fixed for this image)
SPLUNK_BUNDLE_DIR="/usr/lib/splunk-otel-collector/agent-bundle"
SPLUNK_COLLECTD_DIR="/usr/lib/splunk-otel-collector/agent-bundle/run/collectd"
SPLUNK_LISTEN_INTERFACE="0.0.0.0"
SPLUNK_MEMORY_LIMIT_MIB="${SPLUNK_MEMORY_LIMIT_MIB:-900}"

# ---------------------------------------------------------------------------
# 1. Create Storage Account + File Share (for the config file)
# ---------------------------------------------------------------------------
echo ""
echo "==> [1/4] Creating storage account: ${STORAGE_ACCOUNT}"
az storage account create \
  --name "${STORAGE_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --output none

STORAGE_KEY=$(az storage account keys list \
  --resource-group "${RESOURCE_GROUP}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --query "[0].value" \
  --output tsv)

echo "==> [1/4] Creating file share: ${FILE_SHARE}"
az storage share create \
  --name "${FILE_SHARE}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --account-key "${STORAGE_KEY}" \
  --output none

# ---------------------------------------------------------------------------
# 2. Upload collector config to the file share
# ---------------------------------------------------------------------------
echo ""
echo "==> [2/4] Uploading otel-collector-config.yaml to file share"
az storage file upload \
  --share-name "${FILE_SHARE}" \
  --source "${CONFIG_FILE}" \
  --path "otel-collector-config.yaml" \
  --account-name "${STORAGE_ACCOUNT}" \
  --account-key "${STORAGE_KEY}" \
  --output none

# ---------------------------------------------------------------------------
# 3. Deploy the Azure Container Instance
# ---------------------------------------------------------------------------
echo ""
echo "==> [3/4] Creating container instance: ${CONTAINER_NAME}"

az container create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_NAME}" \
  --image "${CONTAINER_IMAGE}" \
  --os-type Linux \
  --cpu 1 \
  --memory 2 \
  --restart-policy Always \
  \
  --ports 4317 4318 13133 6060 9411 \
  --ip-address Public \
  \
  --azure-file-volume-account-name "${STORAGE_ACCOUNT}" \
  --azure-file-volume-account-key "${STORAGE_KEY}" \
  --azure-file-volume-share-name "${FILE_SHARE}" \
  --azure-file-volume-mount-path /etc/otel \
  \
  --command-line "/otelcol --config /etc/otel/otel-collector-config.yaml" \
  \
  --environment-variables \
    SPLUNK_ACCESS_TOKEN="${SPLUNK_ACCESS_TOKEN}" \
    SPLUNK_HEC_TOKEN="${SPLUNK_HEC_TOKEN}" \
    SPLUNK_HEC_URL="${SPLUNK_HEC_URL}" \
    SPLUNK_API_URL="${SPLUNK_API_URL}" \
    SPLUNK_INGEST_URL="${SPLUNK_INGEST_URL}" \
    SPLUNK_TRACE_URL="${SPLUNK_TRACE_URL}" \
    SPLUNK_REALM="${SPLUNK_REALM}" \
    SPLUNK_BUNDLE_DIR="${SPLUNK_BUNDLE_DIR}" \
    SPLUNK_COLLECTD_DIR="${SPLUNK_COLLECTD_DIR}" \
    SPLUNK_LISTEN_INTERFACE="${SPLUNK_LISTEN_INTERFACE}" \
    SPLUNK_MEMORY_LIMIT_MIB="${SPLUNK_MEMORY_LIMIT_MIB}" \
    OTEL_RESOURCE_ATTRIBUTES="deployment.environment=${DEPLOYMENT_ENV}" \
  \
  --output table

# ---------------------------------------------------------------------------
# 4. Show the public IP so we can update OTEL_EXPORTER_OTLP_ENDPOINT
# ---------------------------------------------------------------------------
echo ""
echo "==> [4/4] Fetching container IP address"
COLLECTOR_IP=$(az container show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_NAME}" \
  --query "ipAddress.ip" \
  --output tsv)

echo ""
echo "============================================================"
echo " Splunk OTel Collector deployed successfully!"
echo "============================================================"
echo " Container:    ${CONTAINER_NAME}"
echo " Public IP:    ${COLLECTOR_IP}"
echo ""
echo " Update your Streamlit app's .env / App Service config:"
echo ""
echo "   OTEL_EXPORTER_OTLP_ENDPOINT=http://${COLLECTOR_IP}:4317"
echo ""
echo " Health check:"
echo "   curl http://${COLLECTOR_IP}:13133/"
echo ""
echo " View logs:"
echo "   az container logs --resource-group ${RESOURCE_GROUP} --name ${CONTAINER_NAME} --follow"
echo "============================================================"
