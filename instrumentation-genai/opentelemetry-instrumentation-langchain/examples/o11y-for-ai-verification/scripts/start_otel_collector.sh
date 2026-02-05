#!/bin/bash

# =============================================================================
# Start OpenTelemetry Collector
# =============================================================================
# This script starts the OTEL Collector with the proper configuration
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
ENV_FILE="${PROJECT_ROOT}/config/.env.rc0.template"
OTEL_CONFIG="${PROJECT_ROOT}/bin/otel-config-ai.yaml"
OTEL_COLLECTOR_PATH="${PROJECT_ROOT}/bin/otelcol_darwin_arm64"
LOG_FILE="${PROJECT_ROOT}/otel-collector.log"

echo -e "${BLUE}Starting OpenTelemetry Collector...${NC}\n"

# Load environment variables
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

# Validate collector exists
if [ ! -f "$OTEL_COLLECTOR_PATH" ]; then
    echo -e "${RED}Error: OTEL Collector not found at: $OTEL_COLLECTOR_PATH${NC}"
    exit 1
fi

# Make executable
chmod +x "$OTEL_COLLECTOR_PATH" 2>/dev/null || true

# Set required environment variables for otel-config-ai.yaml
export SPLUNK_LISTEN_INTERFACE="0.0.0.0"
export SPLUNK_TRACE_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"
export SPLUNK_MEMORY_LIMIT_MIB="${SPLUNK_MEMORY_TOTAL_MIB:-512}"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Realm: ${SPLUNK_REALM}"
echo -e "  API URL: ${SPLUNK_API_URL}"
echo -e "  Ingest URL: ${SPLUNK_INGEST_URL}"
echo -e "  Trace URL: ${SPLUNK_TRACE_URL}"
echo -e "  HEC URL: ${SPLUNK_HEC_URL}"
echo -e "  Config: ${OTEL_CONFIG}"
echo -e "  Log File: ${LOG_FILE}\n"

# Check if already running
if pgrep -f "otelcol_darwin_arm64" > /dev/null; then
    echo -e "${YELLOW}Warning: OTEL Collector is already running${NC}"
    echo -e "${YELLOW}Stop it first with: pkill -f otelcol_darwin_arm64${NC}\n"
    read -p "Kill existing collector and restart? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "otelcol_darwin_arm64"
        sleep 2
    else
        exit 1
    fi
fi

# Start collector
echo -e "${GREEN}Starting collector...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop, or run in background with: $0 &${NC}\n"

cd "${PROJECT_ROOT}/bin"
./otelcol_darwin_arm64 --config otel-config-ai.yaml 2>&1 | tee "$LOG_FILE"
