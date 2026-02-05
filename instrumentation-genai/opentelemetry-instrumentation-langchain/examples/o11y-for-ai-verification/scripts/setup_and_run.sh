#!/bin/bash

# =============================================================================
# Alpha Release Testing - Setup and Run Script
# =============================================================================
# This script:
# 1. Loads environment variables from config/.env.rc0.template
# 2. Starts the OpenTelemetry Collector
# 3. Provides instructions for running test applications
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
ENV_FILE="${PROJECT_ROOT}/config/.env.rc0.template"
OTEL_CONFIG="${PROJECT_ROOT}/bin/otel-config-ai.yaml"
OTEL_COLLECTOR_PATH="${PROJECT_ROOT}/bin/otelcol_darwin_arm64"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Alpha Release Testing - Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# =============================================================================
# Step 1: Load Environment Variables
# =============================================================================
echo -e "${YELLOW}[1/3] Loading environment variables...${NC}"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

# Export all variables from the env file
set -a
source "$ENV_FILE"
set +a

echo -e "${GREEN}✓ Environment variables loaded from: $ENV_FILE${NC}\n"

# =============================================================================
# Step 2: Validate OTEL Collector
# =============================================================================
echo -e "${YELLOW}[2/3] Validating OpenTelemetry Collector...${NC}"

if [ ! -f "$OTEL_COLLECTOR_PATH" ]; then
    echo -e "${RED}Error: OTEL Collector not found at: $OTEL_COLLECTOR_PATH${NC}"
    echo -e "${YELLOW}Please download the collector or update OTEL_COLLECTOR_PATH in this script${NC}"
    exit 1
fi

# Make collector executable if it isn't already
if [ ! -x "$OTEL_COLLECTOR_PATH" ]; then
    echo -e "${YELLOW}Making collector executable...${NC}"
    chmod +x "$OTEL_COLLECTOR_PATH"
fi

echo -e "${GREEN}✓ OTEL Collector found: $OTEL_COLLECTOR_PATH${NC}\n"

# =============================================================================
# Step 3: Set Required Environment Variables for OTEL Collector
# =============================================================================
echo -e "${YELLOW}[3/3] Configuring OTEL Collector environment...${NC}"

# Construct Splunk ingest URL from realm
export SPLUNK_INGEST_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace/otlp"
export SPLUNK_MEMORY_TOTAL_MIB="${SPLUNK_MEMORY_TOTAL_MIB:-512}"

echo -e "${GREEN}✓ OTEL Collector configured${NC}"
echo -e "  Realm: ${SPLUNK_REALM}"
echo -e "  Ingest URL: ${SPLUNK_INGEST_URL}"
echo -e "  Config: ${OTEL_CONFIG}\n"

# =============================================================================
# Display Next Steps
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Environment variables are now loaded in this shell session.${NC}\n"

echo -e "${YELLOW}Next Steps:${NC}\n"

echo -e "${BLUE}1. Start the OTEL Collector (in a separate terminal):${NC}"
echo -e "   cd ${PROJECT_ROOT}"
echo -e "   cd ${PROJECT_ROOT}/bin"
echo -e "   export SPLUNK_LISTEN_INTERFACE=\"0.0.0.0\""
echo -e "   export SPLUNK_TRACE_URL=\"https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace\""
echo -e "   export SPLUNK_MEMORY_LIMIT_MIB=\"${SPLUNK_MEMORY_TOTAL_MIB}\""
echo -e "   ./otelcol_darwin_arm64 --config otel-config-ai.yaml\n"

echo -e "${BLUE}2. Run a test application (in this terminal):${NC}"
echo -e "   cd ${PROJECT_ROOT}/tests/apps"
echo -e "   python retail_shop_langchain_app.py"
echo -e "   # or"
echo -e "   python langgraph_travel_planner_app.py\n"

echo -e "${BLUE}3. Run the test framework:${NC}"
echo -e "   cd ${PROJECT_ROOT}/o11y-for-ai-verification"
echo -e "   pytest tests/ui/ -v\n"

echo -e "${YELLOW}Tip: To start the OTEL Collector in the background:${NC}"
echo -e "   ${SCRIPT_DIR}/start_otel_collector.sh\n"

# =============================================================================
# Optional: Start OTEL Collector
# =============================================================================
read -p "Do you want to start the OTEL Collector now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Starting OTEL Collector...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"
    
    export SPLUNK_LISTEN_INTERFACE="0.0.0.0"
    export SPLUNK_TRACE_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"
    export SPLUNK_MEMORY_LIMIT_MIB="${SPLUNK_MEMORY_TOTAL_MIB}"
    
    cd "${PROJECT_ROOT}/bin"
    ./otelcol_darwin_arm64 --config otel-config-ai.yaml
else
    echo -e "\n${GREEN}You can start the collector manually using the commands above.${NC}"
fi
