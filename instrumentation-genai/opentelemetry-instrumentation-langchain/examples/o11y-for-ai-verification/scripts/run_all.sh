#!/bin/bash
################################################################################
# Master Orchestration Script - End-to-End Automation
# 
# Purpose: Complete automation of:
#   1. Environment setup
#   2. OTel Collector startup
#   3. Test application execution (1 iteration per scenario)
#   4. Test framework execution
#   5. Advanced report generation
#
# Usage: ./scripts/run_all.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     🚀 O11y AI Test Framework - Complete Automation 🚀        ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Track start time
START_TIME=$(date +%s)

# Step 0: Verify Virtual Environment Setup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Step 0/5: Verifying Virtual Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VENV_PATH="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found"
    echo ""
    echo "Please run setup first:"
    echo "  ./scripts/setup_venv.sh"
    echo ""
    exit 1
fi

# Activate and verify dependencies
source "$VENV_PATH/bin/activate"

MISSING_DEPS=false
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "❌ Missing: pytest"
    MISSING_DEPS=true
fi

if ! python3 -c "import requests" 2>/dev/null; then
    echo "❌ Missing: requests"
    MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    echo "❌ ERROR: Missing required dependencies"
    echo "   Please run: ./scripts/setup_venv.sh"
    exit 1
fi

echo "✅ Virtual environment verified"
echo ""
sleep 1

# Step 1: Environment Setup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Step 1/5: Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
source "$SCRIPT_DIR/setup_environment.sh"

# Check for Azure OpenAI credentials (primary provider)
if [ -z "$AZURE_OPENAI_ENDPOINT" ] || [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo ""
    echo "❌ ERROR: Azure OpenAI credentials not configured!"
    echo ""
    echo "Azure OpenAI is the PRIMARY LLM provider for tests."
    echo ""
    echo "Option 1: Create config file (recommended)"
    echo "  cp config/azure_openai.env.template config/azure_openai.env"
    echo "  # Edit config/azure_openai.env with your credentials"
    echo "  source scripts/setup_environment.sh"
    echo ""
    echo "Option 2: Set environment variables"
    echo "  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'"
    echo "  export AZURE_OPENAI_API_KEY='your-api-key'"
    echo "  export AZURE_OPENAI_DEPLOYMENT='gpt-4'"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "✅ Environment configured successfully"
sleep 2

# Step 2: Start OTel Collector
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 Step 2/5: Starting OpenTelemetry Collector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"$SCRIPT_DIR/start_collector.sh"

echo ""
echo "✅ Collector started successfully"
echo "   Waiting 5 seconds for collector to be ready..."
sleep 5

# Step 3: Run Test Applications
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Step 3/5: Running Test Applications"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will execute all 10 scenarios, 1 iteration each (10 total runs)"
echo "Estimated time: 8-10 minutes"
echo ""

if "$SCRIPT_DIR/run_apps.sh" all 1; then
    echo ""
    echo "✅ All applications executed successfully"
    echo "   Waiting 30 seconds for telemetry to propagate..."
    sleep 30
else
    echo ""
    echo "⚠️  Some applications failed, but continuing with tests..."
    sleep 10
fi

# Step 4: Run Test Framework
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Step 4/5: Executing Test Framework"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if "$SCRIPT_DIR/run_tests.sh"; then
    TEST_STATUS="✅ All tests passed"
    TEST_EXIT_CODE=0
else
    TEST_STATUS="⚠️  Some tests failed"
    TEST_EXIT_CODE=1
fi

echo ""
echo "$TEST_STATUS"
sleep 2

# Step 5: Generate Advanced Reports
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 5/5: Generating Advanced Reports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "$SCRIPT_DIR/generate_report.py" "$PROJECT_ROOT"

# Calculate total execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Final Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║                  🎉 AUTOMATION COMPLETE! 🎉                    ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Execution Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Environment configured"
echo "  ✓ OTel Collector started"
echo "  ✓ Test applications executed (10 scenarios × 1 iteration)"
echo "  $TEST_STATUS"
echo "  ✓ Advanced reports generated"
echo ""
echo "⏱️  Total execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 Generated Artifacts:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📄 Trace IDs: .trace_ids.json"
echo "  📊 HTML Report: reports/html/latest_report.html"
echo "  📊 JSON Report: reports/json/latest_report.json"
echo "  📋 Test Logs: reports/*.log"
echo ""
echo "🌐 View Report:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  open $PROJECT_ROOT/reports/html/latest_report.html"
echo ""
echo "🛑 Stop Collector:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  $SCRIPT_DIR/stop_collector.sh"
echo ""

exit $TEST_EXIT_CODE
