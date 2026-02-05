#!/bin/bash
################################################################################
# Automated Test Execution Script
# 
# Purpose: Execute all test cases using trace IDs captured from app executions
#
# Usage: ./scripts/run_tests.sh [--test-file TEST_FILE]
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRACE_IDS_FILE="$PROJECT_ROOT/.trace_ids.json"
VENV_PATH="$PROJECT_ROOT/.venv"
REPORT_DIR="$PROJECT_ROOT/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "🧪 Running Automated Tests"
echo "=========================================="
echo ""

# Check and setup virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found"
    echo "   Please run: ./scripts/setup_venv.sh"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Verify critical dependencies are installed
echo "🔍 Verifying dependencies..."
MISSING_DEPS=false

if ! python3 -c "import pytest" 2>/dev/null; then
    echo "❌ ERROR: pytest not installed"
    MISSING_DEPS=true
fi

if ! python3 -c "import requests" 2>/dev/null; then
    echo "❌ ERROR: requests not installed"
    MISSING_DEPS=true
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    echo "❌ ERROR: pyyaml not installed"
    MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ]; then
    echo ""
    echo "❌ ERROR: Missing required dependencies"
    echo "   Please run: ./scripts/setup_venv.sh"
    exit 1
fi

echo "✅ All dependencies verified"
echo ""

# Verify environment
if [ -z "$SPLUNK_ACCESS_TOKEN" ]; then
    echo "❌ ERROR: Environment not configured"
    echo "Please run: source scripts/setup_environment.sh"
    exit 1
fi

# Check if trace IDs file exists
if [ ! -f "$TRACE_IDS_FILE" ]; then
    echo "⚠️  WARNING: No trace IDs file found"
    echo "   Expected: $TRACE_IDS_FILE"
    echo ""
    echo "Options:"
    echo "  1. Run apps first: ./scripts/run_apps.sh"
    echo "  2. Provide trace ID manually: --trace-id=<ID>"
    echo ""
    echo "❌ ERROR: Cannot run tests without trace IDs"
    exit 1
else
    echo "✅ Found trace IDs file: $TRACE_IDS_FILE"
fi

echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Create reports directory
mkdir -p "$REPORT_DIR/html" "$REPORT_DIR/json" "$REPORT_DIR/junit"

# Define test files, their descriptions, and corresponding scenarios
declare -A TEST_FILES=(
    ["test_tc_pi2_foundation_01_to_05.py"]="Foundation Components (TC-PI2-FOUNDATION-01 to 05)"
    ["test_tc_pi2_langgraph_01.py"]="LangGraph Multi-Agent (TC-PI2-LANGGRAPH-01)"
    ["test_tc_pi2_rag_01.py"]="RAG Pipeline (TC-PI2-RAG-01)"
    ["test_tc_pi2_litellm.py"]="LiteLLM Proxy (TC-PI2-LITELLM-01, 02)"
    ["test_tc_pi2_streaming_01.py"]="Streaming TTFT (TC-PI2-STREAMING-01)"
    ["test_tc_pi2_data_01.py"]="Data Validation (TC-PI2-DATA-01)"
)

# Map test files to scenario names in .trace_ids.json
declare -A TEST_SCENARIOS=(
    ["test_tc_pi2_foundation_01_to_05.py"]="multi_agent_retail"
    ["test_tc_pi2_langgraph_01.py"]="langgraph_workflow"
    ["test_tc_pi2_rag_01.py"]="rag_pipeline"
    ["test_tc_pi2_litellm.py"]="litellm_proxy"
    ["test_tc_pi2_streaming_01.py"]="streaming_ttft"
    ["test_tc_pi2_data_01.py"]="multi_provider_edge_cases"
)

# Function to get trace ID for a specific scenario
get_trace_id() {
    local scenario=$1
    python3 -c "
import json
import sys
try:
    with open('$TRACE_IDS_FILE') as f:
        data = json.load(f)
    scenarios = data.get('scenarios', {})
    trace_ids = scenarios.get('$scenario', [])
    if trace_ids and trace_ids[0]:
        print(trace_ids[0])
    else:
        sys.exit(1)
except Exception as e:
    sys.stderr.write(f'Error reading trace ID: {e}\n')
    sys.exit(1)
"
}

TOTAL_TESTS=${#TEST_FILES[@]}
CURRENT=0
PASSED=0
FAILED=0

echo "=========================================="
echo "📋 Test Execution Plan"
echo "=========================================="
echo "Total test suites: $TOTAL_TESTS"
echo "Report directory: $REPORT_DIR"
echo ""

# Run each test file
for test_file in "${!TEST_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    description="${TEST_FILES[$test_file]}"
    scenario="${TEST_SCENARIOS[$test_file]}"
    
    echo ""
    echo "=========================================="
    echo "🧪 Test Suite $CURRENT/$TOTAL_TESTS"
    echo "=========================================="
    echo "File: $test_file"
    echo "Description: $description"
    echo "Scenario: $scenario"
    
    # Get trace ID for this specific scenario
    if [ -n "$scenario" ]; then
        TRACE_ID=$(get_trace_id "$scenario")
        if [ $? -eq 0 ] && [ -n "$TRACE_ID" ]; then
            TRACE_ID_PARAM="--trace-id=$TRACE_ID"
            echo "Trace ID: $TRACE_ID"
        else
            echo "⚠️  WARNING: No trace ID found for scenario '$scenario'"
            TRACE_ID_PARAM=""
        fi
    else
        echo "⚠️  WARNING: No scenario mapping for test file"
        TRACE_ID_PARAM=""
    fi
    
    echo ""
    
    # Run pytest with comprehensive reporting
    if pytest "tests/$test_file" \
        $TRACE_ID_PARAM \
        -v \
        --tb=short \
        --log-cli-level=INFO \
        --log-file="$REPORT_DIR/test_execution.log" \
        --log-file-level=DEBUG \
        --html="$REPORT_DIR/html/${test_file%.py}_${TIMESTAMP}.html" \
        --self-contained-html \
        --junit-xml="$REPORT_DIR/junit/${test_file%.py}_${TIMESTAMP}.xml" \
        --json-report \
        --json-report-file="$REPORT_DIR/json/${test_file%.py}_${TIMESTAMP}.json" \
        --alluredir="$REPORT_DIR/allure-results" \
        2>&1 | tee -a "$REPORT_DIR/${test_file%.py}_${TIMESTAMP}.log"; then
        
        echo ""
        echo "✅ PASSED: $description"
        PASSED=$((PASSED + 1))
    else
        echo ""
        echo "❌ FAILED: $description"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "📊 Test Execution Summary"
echo "=========================================="
echo "Total test suites: $TOTAL_TESTS"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL_TESTS)*100}")%"
echo ""
echo "Reports generated:"
echo "  - HTML: $REPORT_DIR/html/"
echo "  - JSON: $REPORT_DIR/json/"
echo "  - JUnit: $REPORT_DIR/junit/"
echo "  - Logs: $REPORT_DIR/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed - review reports for details"
    exit 1
fi
