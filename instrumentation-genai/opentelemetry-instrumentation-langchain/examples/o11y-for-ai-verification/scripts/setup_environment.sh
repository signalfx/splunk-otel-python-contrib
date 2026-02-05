#!/bin/bash
################################################################################
# Environment Setup Script for O11y AI Test Framework
# 
# Purpose: Configure all required environment variables for:
#   - Splunk Observability Cloud (RC0 realm)
#   - OpenTelemetry Collector
#   - Test Applications
#   - Test Framework
#
# Usage: source scripts/setup_environment.sh
################################################################################

set -e

echo "=========================================="
echo "🔧 O11y AI Test Framework - Environment Setup"
echo "=========================================="

# Get the directory where this script is located (works for both source and direct execution)
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
# PROJECT_ROOT is the parent of scripts/ directory (o11y-for-ai-verification)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"

# Debug: Show resolved paths
echo "📂 Config directory: $CONFIG_DIR"

# Load Azure OpenAI credentials from config directory
AZURE_CONFIG_FILE="$CONFIG_DIR/azure_openai.env"
RC0_CONFIG_FILE="$CONFIG_DIR/.env.rc0.template"

# Try to load from azure_openai.env first (user-created config)
if [ -f "$AZURE_CONFIG_FILE" ]; then
    echo "📁 Loading Azure OpenAI config from: $AZURE_CONFIG_FILE"
    set -a
    source "$AZURE_CONFIG_FILE"
    set +a
# Fallback to rc0 template if azure_openai.env doesn't exist
elif [ -f "$RC0_CONFIG_FILE" ]; then
    echo "📁 Loading config from: $RC0_CONFIG_FILE"
    set -a
    source "$RC0_CONFIG_FILE"
    set +a
else
    echo "⚠️  No config file found. Using environment variables only."
fi

# Splunk Observability Cloud Configuration (RC0)
export SPLUNK_ACCESS_TOKEN="${SPLUNK_ACCESS_TOKEN:-WSGJU11qd4hwf0OBECQeWg}"
export SPLUNK_REALM="${SPLUNK_REALM:-rc0}"
export SPLUNK_API_URL="${SPLUNK_API_URL:-https://api.rc0.signalfx.com}"
export SPLUNK_INGEST_URL="${SPLUNK_INGEST_URL:-https://ingest.rc0.signalfx.com}"
export SPLUNK_HEC_URL="${SPLUNK_HEC_URL:-https://http-inputs-o11y-cosmicbat.splunkcloud.com:443/services/collector}"
export SPLUNK_HEC_TOKEN="${SPLUNK_HEC_TOKEN:-3D43DC7C-BB70-4C7C-B9FC-895AD66C67D7}"
export SPLUNK_MEMORY_TOTAL_MIB="${SPLUNK_MEMORY_TOTAL_MIB:-512}"

# OpenTelemetry Configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE="DELTA"
export OTEL_LOGS_EXPORTER="otlp"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
export OTEL_SERVICE_NAME="alpha-test-unified-app"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=alpha-test-rc0,test.framework=o11y-ai"

# GenAI Instrumentation Configuration
export OTEL_INSTRUMENTATION_GENAI_ENABLE="true"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MODE="SPAN_AND_EVENT"
export OTEL_INSTRUMENTATION_GENAI_EMITTERS="span_metric_event"
export OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS="deepeval(LLMInvocation(bias,toxicity,hallucination,relevance,sentiment),AgentInvocation(bias,toxicity,hallucination,relevance,sentiment))"

# LangChain Specific Configuration
export OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT="true"

# Test Framework Configuration
export TEST_ENVIRONMENT="rc0"
export TEST_TIMEOUT="300"
export TEST_RETRY_COUNT="3"
export TEST_PARALLEL_WORKERS="4"

# Reporting Configuration
export REPORT_OUTPUT_DIR="./reports"
export REPORT_FORMAT="html,json"
export REPORT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Optional: DeepEval Configuration
export DEEPEVAL_TELEMETRY_OPT_OUT="YES"
export DEEPEVAL_FILE_SYSTEM="READ_ONLY"

# DeepEval Judge Model Configuration - Use Azure OpenAI
# These DEEPEVAL_LLM_* variables configure the judge model for evaluation metrics
# (bias, toxicity, hallucination, relevance, sentiment)
# Note: For Azure OpenAI with LiteLLM, base_url should be just the endpoint
# LiteLLM constructs the full URL internally for Azure provider
if [ -n "$AZURE_OPENAI_ENDPOINT" ] && [ -n "$AZURE_OPENAI_API_KEY" ]; then
    export DEEPEVAL_LLM_BASE_URL="${AZURE_OPENAI_ENDPOINT}"
    export DEEPEVAL_LLM_MODEL="${AZURE_OPENAI_DEPLOYMENT:-gpt-4}"
    export DEEPEVAL_LLM_PROVIDER="azure"
    # Azure OpenAI requires api-version parameter
    export AZURE_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-08-01-preview}"
    export DEEPEVAL_LLM_API_KEY="$AZURE_OPENAI_API_KEY"
    echo "✅ DeepEval configured to use Azure OpenAI as judge model"
fi

# Circuit API Configuration (for Cisco network/VPN access)
# All 5 variables must be set to use Circuit API, otherwise falls back to OpenAI
export CIRCUIT_BASE_URL="${CIRCUIT_BASE_URL:-https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini}"
export CIRCUIT_TOKEN_URL="${CIRCUIT_TOKEN_URL:-https://id.cisco.com/oauth2/default/v1/token}"
export CIRCUIT_CLIENT_ID="${CIRCUIT_CLIENT_ID:-0oarbtuuh0w0QsPrJ5d7}"
export CIRCUIT_CLIENT_SECRET="${CIRCUIT_CLIENT_SECRET:-7sXq_UC3xFBVP5UWdj53wIxQqLax402er-UPUM-0FJIUL7kvlMSICYHTj9X7uhPT}"
export CIRCUIT_APP_KEY="${CIRCUIT_APP_KEY:-egai-prd-other-123028255-coding-1762238492681}"

# Verify critical variables are set
echo ""
echo "✅ Environment Variables Configured:"
echo "   - SPLUNK_REALM: $SPLUNK_REALM"
echo "   - SPLUNK_API_URL: $SPLUNK_API_URL"
echo "   - OTEL_EXPORTER_OTLP_ENDPOINT: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "   - OTEL_SERVICE_NAME: $OTEL_SERVICE_NAME"
echo "   - TEST_ENVIRONMENT: $TEST_ENVIRONMENT"
echo ""

# Check if Azure OpenAI credentials are set (required - primary provider)
if [ -z "$AZURE_OPENAI_ENDPOINT" ] || [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: Azure OpenAI credentials not fully configured!"
    echo "   This is the PRIMARY LLM provider for tests."
    echo ""
    echo "   Option 1: Create config file (recommended)"
    echo "     cp config/azure_openai.env.template config/azure_openai.env"
    echo "     # Edit config/azure_openai.env with your credentials"
    echo ""
    echo "   Option 2: Set environment variables"
    echo "     export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'"
    echo "     export AZURE_OPENAI_API_KEY='your-api-key'"
    echo "     export AZURE_OPENAI_DEPLOYMENT='gpt-4'"
    echo ""
else
    echo "✅ Azure OpenAI configured:"
    echo "   - Endpoint: $AZURE_OPENAI_ENDPOINT"
    echo "   - Deployment: ${AZURE_OPENAI_DEPLOYMENT:-gpt-4}"
    echo ""
fi

# Check if OpenAI API key is set (optional fallback)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "ℹ️  INFO: OpenAI API key also configured (fallback provider)"
    echo ""
fi

# Check if Circuit API is configured
if [ -n "$CIRCUIT_BASE_URL" ] && [ -n "$CIRCUIT_TOKEN_URL" ] && [ -n "$CIRCUIT_CLIENT_ID" ] && [ -n "$CIRCUIT_CLIENT_SECRET" ] && [ -n "$CIRCUIT_APP_KEY" ]; then
    echo "✅ Circuit API configured (requires Cisco network/VPN):"
    echo "   - Base URL: $CIRCUIT_BASE_URL"
    echo "   - Token URL: $CIRCUIT_TOKEN_URL"
    echo ""
fi

echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start OTel Collector: ./scripts/start_collector.sh"
echo "  2. Run test apps: ./scripts/run_apps.sh"
echo "  3. Execute tests: ./scripts/run_tests.sh"
echo "  4. Or run all: ./scripts/run_all.sh"
echo ""
