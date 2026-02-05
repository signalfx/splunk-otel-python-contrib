#!/bin/bash

# =============================================================================
# Switch Realm Script
# =============================================================================
# This script switches between different Splunk realms (rc0, us1, lab0, mon0)
# by loading the appropriate environment configuration
# =============================================================================

# Note: Don't use 'set -e' when script is meant to be sourced
# as it will close the terminal on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory - handle both sourced and executed cases
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    # Fallback if BASH_SOURCE is not available
    SCRIPT_DIR="$(pwd)/scripts"
    PROJECT_ROOT="$(pwd)"
fi

# Usage function
usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  $0 <realm>"
    echo -e ""
    echo -e "${BLUE}Available realms:${NC}"
    echo -e "  ${GREEN}rc0${NC}   - RC0 environment"
    echo -e "  ${GREEN}us1${NC}   - US1 production environment"
    echo -e "  ${GREEN}lab0${NC}  - LAB0 environment"
    echo -e "  ${GREEN}mon0${NC}  - MON0 environment"
    echo -e ""
    echo -e "${BLUE}Example:${NC}"
    echo -e "  $0 rc0"
    echo -e "  source $0 rc0  ${YELLOW}# Use 'source' to export variables to current shell${NC}"
    return 1 2>/dev/null || exit 1
}

# Check arguments
if [ $# -ne 1 ]; then
    usage
fi

REALM=$1

# Validate realm
case $REALM in
    rc0|us1|lab0|mon0)
        ;;
    *)
        echo -e "${RED}Error: Invalid realm '$REALM'${NC}"
        echo -e "${YELLOW}Valid realms: rc0, us1, lab0, mon0${NC}"
        return 1 2>/dev/null || exit 1
        ;;
esac

# Environment file
ENV_FILE="${PROJECT_ROOT}/config/.env.${REALM}.template"

# Check if file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    return 1 2>/dev/null || exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Switching to ${GREEN}${REALM}${BLUE} realm${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Load environment variables
echo -e "${YELLOW}Loading environment variables from:${NC}"
echo -e "  $ENV_FILE\n"

set -a
source "$ENV_FILE" || {
    echo -e "${RED}Error: Failed to load environment file${NC}"
    set +a
    return 1 2>/dev/null || exit 1
}
set +a

# Display key configuration
echo -e "${GREEN}✓ Environment loaded successfully${NC}\n"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Realm: ${GREEN}${SPLUNK_REALM}${NC}"
echo -e "  API URL: ${SPLUNK_API_URL:-Not set}"
echo -e "  Ingest URL: ${SPLUNK_INGEST_URL:-Not set}"
echo -e "  HEC URL: ${SPLUNK_HEC_URL}"
echo -e "  Service Name: ${OTEL_SERVICE_NAME}"
echo -e "  Environment: ${OTEL_RESOURCE_ATTRIBUTES:-Not set}\n"

# Set additional required variables for otel-config-ai.yaml
export SPLUNK_INGEST_URL="${SPLUNK_INGEST_URL:-https://ingest.${SPLUNK_REALM}.signalfx.com}"
export SPLUNK_API_URL="${SPLUNK_API_URL:-https://api.${SPLUNK_REALM}.signalfx.com}"
export SPLUNK_TRACE_URL="https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace"
export SPLUNK_LISTEN_INTERFACE="0.0.0.0"
export SPLUNK_MEMORY_LIMIT_MIB="${SPLUNK_MEMORY_TOTAL_MIB:-512}"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Start OTEL Collector: ${BLUE}./scripts/start_otel_collector.sh${NC}"
echo -e "  2. Run test app: ${BLUE}cd tests/apps && python retail_shop_langchain_app.py${NC}"
echo -e "  3. Run tests: ${BLUE}cd o11y-for-ai-verification && pytest tests/ui/ -v${NC}\n"

echo -e "${GREEN}Environment variables are now loaded!${NC}"
echo -e "${YELLOW}Note: If you didn't use 'source', variables are only set in this script's context.${NC}"
echo -e "${YELLOW}Use: ${BLUE}source $0 $REALM${YELLOW} to export to your current shell.${NC}\n"
